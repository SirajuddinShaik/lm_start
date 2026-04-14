"""
Adaptive Validator

Tests vLLM configs with intelligent retry logic:
1. Try config
2. If fails, analyze error
3. Try optimizations first (keep capacity)
4. Only reduce capacity if optimizations fail
5. Never reduce max_model_len unless absolutely necessary
"""

import json
import subprocess
import time
import requests
import signal
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from vllm_flag_validator import validate_vllm_flags


@dataclass
class ValidationResult:
    """Result of config validation."""

    success: bool
    config: Dict[str, Any]
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    startup_time: Optional[float] = None
    port: int = 8000
    process: Optional[subprocess.Popen] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "startup_time": self.startup_time,
            "port": self.port,
        }


class AdaptiveValidator:
    """
    Validates vLLM configs with adaptive retry.

    Strategy:
    1. Try calculated config
    2. If OOM: Try memory optimizations (chunked_prefill, block_size, util)
    3. If still OOM: Reduce max_num_seqs (keep max_model_len!)
    4. If still OOM: Reduce max_num_batched_tokens
    5. Last resort: Reduce max_model_len
    """

    def __init__(
        self,
        model_dir: Path,
        model_id: str,
        venv_python: Path,
        hf_home: str = "/data/.cache/huggingface",
        cuda_home: str = "/usr/local/cuda-12.9",
        startup_timeout: int = 1800,
        verbose: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.model_id = model_id
        self.venv_python = Path(venv_python)
        self.hf_home = hf_home
        self.cuda_home = cuda_home
        self.startup_timeout = startup_timeout
        self.verbose = verbose
        self.reasoning_log: List[str] = []

        self.visible_devices = self._load_visible_devices()

        self.attempt_number = 0
        self.base_port = 8000

    def _load_visible_devices(self) -> str:
        device_config_path = self.model_dir / ".llm-context" / "model-context" / "device_config.json"
        if device_config_path.exists():
            try:
                with open(device_config_path) as f:
                    device_config = json.load(f)
                gpus = device_config.get("gpus", {})
                return gpus.get("visible_devices", "0,1,2,3,4,5,6,7")
            except Exception:
                pass

        try:
            from core.config import get_config

            config = get_config()
            return config.device.gpu.visible_devices
        except Exception:
            return "0,1,2,3,4,5,6,7"

    def log(self, message: str, level: str = "info"):
        """Log message with reasoning."""
        self.reasoning_log.append(f"[{level.upper()}] {message}")
        if self.verbose:
            prefix = "[AdaptiveValidator]"
            print(f"{prefix} {message}")

    def validate(
        self,
        config: Dict[str, Any],
        max_retries: int = 10,
    ) -> Tuple[ValidationResult, Dict[str, Any]]:
        """
        Validate config with adaptive retry.

        Returns:
            Tuple of (final_result, final_config)
            final_config may be modified from input
        """
        self.reasoning_log = []
        current_config = config.copy()

        self.log(f"Starting validation with max {max_retries} retries")
        self.log(
            f"Initial config: max_seqs={current_config.get('max_num_seqs')}, "
            f"max_model_len={current_config.get('max_model_len', 0):,}"
        )

        for attempt in range(1, max_retries + 1):
            self.attempt_number = attempt
            port = self.base_port + attempt

            self.log(f"\n--- Attempt {attempt}/{max_retries} ---")
            self.log(
                f"Testing: max_seqs={current_config.get('max_num_seqs')}, "
                f"batch_tokens={current_config.get('max_num_batched_tokens', 0):,}, "
                f"model_len={current_config.get('max_model_len', 0):,}"
            )

            # Test this config
            result = self._test_config(current_config, port)

            if result.success:
                self.log(
                    f"✓ SUCCESS! Server ready in {result.startup_time:.1f}s", "success"
                )
                return result, current_config

            # Failed - analyze and adapt
            self.log(f"✗ FAILED: {result.error_type}", "error")

            if result.error_type == "cuda_oom":
                new_config = self._handle_oom(current_config, attempt)
                if new_config is None:
                    self.log("Exhausted OOM recovery strategies", "error")
                    break
                current_config = new_config
            elif result.error_type == "context_too_large":
                # Context length issue - must reduce max_model_len
                new_config = self._reduce_context(current_config)
                if new_config is None:
                    break
                current_config = new_config
            else:
                # Other error - try a few general fixes
                new_config = self._handle_general_error(
                    current_config, result.error_type
                )
                if new_config is None:
                    break
                current_config = new_config

        # Exhausted retries
        self.log(f"\nExhausted {max_retries} attempts without success", "error")
        return ValidationResult(success=False, config=current_config), current_config

    def _test_config(self, config: Dict[str, Any], port: int) -> ValidationResult:
        """Test a single config by starting vLLM."""
        result = ValidationResult(success=False, config=config, port=port)

        # Build command
        cmd = self._build_command(config, port)

        env = os.environ.copy()
        tp = config.get("tensor_parallel_size", 8)
        devices = self.visible_devices.split(",")[:tp]
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        env["VLLM_ATTENTION_BACKEND"] = config.get("attention_backend", "FLASHINFER")
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        env["HF_HOME"] = self.hf_home
        env["CUDA_HOME"] = self.cuda_home

        # Run
        process = None
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid,
            )

            start_time = time.time()

            # Wait for health check
            while time.time() - start_time < self.startup_timeout:
                try:
                    resp = requests.get(f"http://localhost:{port}/health", timeout=5)
                    if resp.status_code == 200:
                        result.success = True
                        result.startup_time = time.time() - start_time
                        result.process = process
                        break
                except:
                    pass

                # Check if process died
                if process.poll() is not None:
                    stderr = process.stderr.read().decode() if process.stderr else ""
                    result.error_message = stderr
                    result.error_type = self._classify_error(stderr)
                    break

                time.sleep(5)
            else:
                # Timeout
                result.error_type = "startup_timeout"
                result.error_message = (
                    f"Server didn't start within {self.startup_timeout}s"
                )
                self._kill_process(process)

        except Exception as e:
            result.error_type = "exception"
            result.error_message = str(e)
            if process:
                self._kill_process(process)

        return result

    def _build_command(self, config: Dict[str, Any], port: int) -> List[str]:
        flag_config = {}

        if config.get("tensor_parallel_size", 1) > 1:
            flag_config["tensor_parallel_size"] = config["tensor_parallel_size"]

        if config.get("max_model_len"):
            flag_config["max_model_len"] = config["max_model_len"]

        if config.get("max_num_seqs"):
            flag_config["max_num_seqs"] = config["max_num_seqs"]

        if config.get("max_num_batched_tokens"):
            flag_config["max_num_batched_tokens"] = config["max_num_batched_tokens"]

        if config.get("gpu_memory_utilization"):
            flag_config["gpu_memory_utilization"] = config["gpu_memory_utilization"]

        if config.get("kv_cache_dtype"):
            flag_config["kv_cache_dtype"] = config["kv_cache_dtype"]

        if config.get("dtype"):
            flag_config["dtype"] = config["dtype"]

        if config.get("enable_chunked_prefill"):
            flag_config["enable_chunked_prefill"] = True

        if config.get("enable_prefix_caching"):
            flag_config["enable_prefix_caching"] = True

        if config.get("enforce_eager"):
            flag_config["enforce_eager"] = True

        if config.get("trust_remote_code"):
            flag_config["trust_remote_code"] = True

        if config.get("enable_expert_parallel"):
            flag_config["enable_expert_parallel"] = True

        validated_args = validate_vllm_flags(
            flag_config,
            model_dir=str(self.model_dir) if hasattr(self, "model_dir") else None,
            vllm_binary=str(self.venv_python).replace("/bin/python", "/bin/vllm")
            if hasattr(self, "venv_python")
            else None,
        )

        return [
            str(self.venv_python),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_id,
            "--port",
            str(port),
        ] + validated_args

    def _classify_error(self, stderr: str) -> str:
        """Classify error from stderr."""
        stderr_lower = stderr.lower()

        if (
            "out of memory" in stderr_lower
            or "oom" in stderr_lower
            or "cudaerror" in stderr_lower
        ):
            return "cuda_oom"
        elif "max_position_embeddings" in stderr_lower or "context" in stderr_lower:
            return "context_too_large"
        elif "trust_remote_code" in stderr_lower:
            return "remote_code"
        elif "gguf" in stderr_lower:
            return "incompatible_format"
        elif "module" in stderr_lower and "not found" in stderr_lower:
            return "missing_module"
        else:
            return "unknown"

    def _handle_oom(
        self, config: Dict[str, Any], attempt: int
    ) -> Optional[Dict[str, Any]]:
        """
        Handle OOM with adaptive strategy.

        The agent's strategy:
        1. Try optimizations first (chunked_prefill, block_size, util) - keeps all features
        2. Reduce max_num_seqs (concurrent users) - keeps max_model_len intact
        3. Reduce batch tokens (less prefill parallelism) - keeps capacity
        4. Last resort: reduce max_model_len (limits model capability)

        Reasoning: We want to preserve full context capability as long as possible.
        Reducing concurrent users is better than reducing per-user context.
        """
        new_config = config.copy()
        current_seqs = new_config.get("max_num_seqs", 16)
        current_batch = new_config.get("max_num_batched_tokens", 32768)
        current_util = new_config.get("gpu_memory_utilization", 0.9)

        # Phase 1: Try optimizations (attempts 1-3)
        # These don't reduce capability, just use memory more efficiently
        if attempt == 1:
            self.log(
                "[REASONING] OOM on first attempt. Strategy: Try memory optimizations first"
            )
            self.log("  Options: chunked_prefill, reduce GPU util, adjust block_size")

            # Enable chunked prefill if not already - interleaves prefill/decode
            if not new_config.get("enable_chunked_prefill", False):
                new_config["enable_chunked_prefill"] = True
                self.log(
                    "[ACTION] Enabled chunked_prefill - allows memory sharing between prefill/decode"
                )
                return new_config

            # Reduce GPU utilization - leaves more headroom
            if current_util > 0.85:
                new_config["gpu_memory_utilization"] = 0.85
                self.log(
                    f"[ACTION] Reduced gpu_memory_utilization: {current_util} → 0.85 (more headroom)"
                )
                return new_config

        if attempt == 2:
            # Block size affects fragmentation - larger blocks = less overhead
            if new_config.get("block_size", 16) < 32:
                new_config["block_size"] = 32
                self.log(
                    "[ACTION] Increased block_size: 16 → 32 (reduces KV fragmentation)"
                )
                return new_config

        # Phase 2: Reduce concurrent users (attempts 3-6)
        # Better to serve fewer users with full context than many users with limited context
        if current_seqs > 4 and attempt <= 6:
            # Reduce by 25% each time (exponential backoff)
            reduction = max(2, int(current_seqs * 0.25))
            new_seqs = max(4, current_seqs - reduction)
            new_config["max_num_seqs"] = new_seqs
            self.log(
                f"[REASONING] Optimizations insufficient. Reducing concurrent users."
            )
            self.log(
                f"[REASONING] Fewer users with full context > Many users with limited context"
            )
            self.log(f"[ACTION] Reducing max_num_seqs: {current_seqs} → {new_seqs}")
            return new_config

        # Phase 3: Reduce batch tokens (attempts 7-8)
        # Smaller batches = less peak memory during prefill
        if current_batch > 8192 and attempt <= 8:
            new_batch = max(8192, current_batch // 2)
            new_config["max_num_batched_tokens"] = new_batch
            self.log(
                f"[REASONING] User capacity already reduced. Now reducing batch parallelism."
            )
            self.log(
                f"[REASONING] Smaller batches = less peak memory, but lower throughput"
            )
            self.log(
                f"[ACTION] Reducing max_num_batched_tokens: {current_batch:,} → {new_batch:,}"
            )
            return new_config

        # Phase 4: Last resort - reduce max_model_len
        # Only do this when all else fails - it limits the model's capability
        self.log("[REASONING] All optimizations exhausted.", "warning")
        self.log(
            "[REASONING] Final option: Reduce max_model_len (limits context capability)",
            "warning",
        )
        return self._reduce_context(new_config)

    def _reduce_context(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Reduce max_model_len as last resort."""
        new_config = config.copy()
        current = new_config.get("max_model_len", 262144)

        # Reduction levels
        levels = [196608, 131072, 98304, 65536, 49152, 32768, 16384, 8192, 4096]

        for level in levels:
            if level < current:
                new_config["max_model_len"] = level
                self.log(
                    f"Reducing max_model_len: {current:,} → {level:,} (LAST RESORT)",
                    "warning",
                )
                return new_config

        # Can't reduce further
        return None

    def _handle_general_error(
        self, config: Dict[str, Any], error_type: str
    ) -> Optional[Dict[str, Any]]:
        """Handle non-OOM errors."""
        new_config = config.copy()

        if error_type == "remote_code":
            new_config["trust_remote_code"] = True
            self.log("Enabling trust_remote_code")
            return new_config

        elif error_type == "incompatible_format":
            new_config["load_format"] = "safetensors"
            self.log("Setting load_format to safetensors")
            return new_config

        # Unknown error - can't fix
        return None

    def _kill_process(self, process: Optional[subprocess.Popen]):
        """Kill process cleanly."""
        if not process:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                pass

    def get_reasoning(self) -> str:
        """Get all reasoning logs."""
        return "\n".join(self.reasoning_log)
