"""
Experiment Agent - Max-Context-First vLLM Configuration Optimization

This agent implements the max-context-first optimization strategy:
1. Start with max_position_embeddings from model config
2. Test configurations with aggressive batching
3. Only reduce context after exhausting all batch parameters
4. Benchmark every working configuration
5. Return the best configuration at or near max context

Key files:
- model_info.json: Model metadata
- device_config.json: GPU configuration
- config.json: HuggingFace model config
- .runs/: Test run results
- model_knowledge_base.json: Verified configurations
"""

import json
import os
import re
import signal
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .base import BaseAgent, AgentResult, AgentStatus

# Import new utilities
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from lm_start.utils.theoretical_calculator import (
    TheoreticalCalculator,
    HardwareConfig,
    ModelConfig,
    calculate_from_model_info,
    get_model_size_from_id,
    detect_model_capabilities,
)
from lm_start.utils.model_utils import (
    get_model_cache_path,
    load_vllm_extracted_config,
    VLLMExtractedConfig,
)
from lm_start.utils.benchmark_runner import BenchmarkRunner, save_benchmark_results
from lm_start.utils.vllm_flag_selector import VLLMFlagSelector, OptimizationProfile
from lm_start.utils.profile_config_generator import ProfileConfigGenerator
from lm_start.vllm_flag_validator import validate_vllm_flags


class ExperimentAgent(BaseAgent):
    """
    Agent that optimizes vLLM configuration using max-context-first strategy.

    Algorithm:
    1. Load model config and calculate theoretical requirements
    2. Start with max_position_embeddings as target context
    3. Generate configs with reduction priority queue:
       - First reduce max_num_batched_tokens
       - Then reduce max_num_seqs
       - Then adjust gpu_memory_utilization
       - Then try different attention backends
       - ONLY THEN reduce max_model_len
    4. Benchmark every working configuration
    5. Select best config based on benchmark results
    """

    # These will be overridden from config in __init__
    VLLM_STARTUP_TIMEOUT = 1800  # 30 minutes for large model loading
    HEALTH_CHECK_INTERVAL = 5

    # Default reduction priority - loaded from flag_knowledge_base.yaml in __init__
    DEFAULT_REDUCTION_PRIORITY = []

    def __init__(
        self,
        model_dir: str,
        max_retries: Optional[int] = None,
        verbose: bool = False,
        use_llm: bool = True,
        hf_home: Optional[str] = None,
        enable_benchmarking: bool = True,
        quick_mode: bool = False,
    ):
        super().__init__("ExperimentAgent", model_dir, max_retries or 10, verbose)
        self.model_dir = Path(model_dir)
        self.use_llm = use_llm
        self.enable_benchmarking = enable_benchmarking
        self.quick_mode = quick_mode

        # Load from config
        self._load_config()

        # Override with passed values
        if hf_home:
            self.hf_home = hf_home

        self.runs_dir = self.model_dir / ".runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.kb_path = self.model_dir / "model_knowledge_base.json"

        # Context
        self.model_info: Dict = {}
        self.device_config: Dict = {}
        self.model_config: Dict = {}  # Actual HF model config
        self.kb: Dict = {}

        # Hardware - defaults, will be overridden by device_config.json
        self.gpu_count = 8
        self.gpu_memory_gb = 80
        self.gpu_name = "unknown"
        self.cuda_devices = "0,1,2,3,4,5,6,7"
        self.cuda_home = "/usr/local/cuda-12.9"
        self.vllm_port = 8000

        # Model
        self.model_id = ""
        self.max_position_embeddings = 32768
        self.model_params_b = 7.0
        self.is_moe = False
        self.is_reasoning = False
        self.supports_tools = False

        # Results
        self.working_configs: List[Dict] = []
        self.failed_configs: List[Dict] = []
        self.benchmark_results: List[Dict] = []
        self.current_context_target = 0

        # Calculator
        self.calculator: Optional[TheoreticalCalculator] = None
        self.hardware: Optional[HardwareConfig] = None

        # Reduction priority - will be loaded from config
        self.reduction_priority: List[Dict] = []

    def _load_config(self):
        """Load configuration from system.yaml."""
        try:
            from core.config import get_config

            config = get_config()

            self.hf_home = config.paths.hf_home
            self.cuda_home = config.device.cuda.home
            self.vllm_port = config.vllm_defaults.port
            self.max_retries = config.experiment.max_runs
            self.VLLM_STARTUP_TIMEOUT = config.monitoring.startup_timeout
            self.HEALTH_CHECK_INTERVAL = config.monitoring.health_check_interval

            # Try to load reduction priority from flag_knowledge_base.yaml
            kb_path = (
                Path(__file__).parent.parent / "configs" / "flag_knowledge_base.yaml"
            )
            if kb_path.exists():
                import yaml

                with open(kb_path) as f:
                    kb = yaml.safe_load(f)
                if "reduction_priority" in kb:
                    self.reduction_priority = kb["reduction_priority"]

        except Exception:
            # Use defaults if config loading fails
            self.hf_home = os.environ.get("HF_HOME", "/data/.cache/huggingface")
            self.cuda_home = "/usr/local/cuda-12.9"
            self.vllm_port = 8000
            # Fallback reduction priority
            self.reduction_priority = [
                {
                    "name": "max_num_batched_tokens",
                    "values": [65536, 32768, 16384, 8192, 4096, 2048],
                },
                {"name": "max_num_seqs", "values": [256, 128, 64, 32, 16]},
                {"name": "enable_chunked_prefill", "values": [True, False]},
                {
                    "name": "gpu_memory_utilization",
                    "values": [0.95, 0.90, 0.85, 0.80, 0.75],
                },
                {
                    "name": "attention_backend",
                    "values": ["FLASHINFER", "FLASH_ATTN", "XFORMERS"],
                },
            ]

    def _load_all_context(self):
        """Load all context files."""
        # 1. Load model_info.json
        model_info_path = (
            self.model_dir / ".llm-context" / "model-context" / "model_info.json"
        )
        if model_info_path.exists():
            with open(model_info_path) as f:
                self.model_info = json.load(f)
            self.model_id = self.model_info.get("model_id", "")
            self.log(f"Model ID: {self.model_id}")

        # 2. Load device_config.json
        device_config_path = (
            self.model_dir / ".llm-context" / "model-context" / "device_config.json"
        )
        if device_config_path.exists():
            with open(device_config_path) as f:
                self.device_config = json.load(f)
            gpus = self.device_config.get("gpus", {})
            self.gpu_count = gpus.get("count", 8)
            self.gpu_memory_gb = gpus.get("memory_gb_per_gpu", 141)
            self.gpu_name = gpus.get("name", "H200")
            self.cuda_devices = gpus.get("visible_devices", "0,1,2,3,4,5,6,7")
            self.log(
                f"GPUs: {self.gpu_count} x {self.gpu_name} ({self.gpu_memory_gb}GB)"
            )

        # 3. Load actual model config from HF cache
        self._load_model_config_from_hf()

        # 4. Load knowledge base
        if self.kb_path.exists():
            with open(self.kb_path) as f:
                self.kb = json.load(f)

        # 5. Initialize calculator
        self._init_calculator()

        # 6. Extract model properties
        self._extract_model_properties()

    def _load_model_config_from_hf(self):
        """Load the actual model config.json from HuggingFace cache."""
        if not self.model_id:
            return

        cache_id = self.model_id.replace("/", "--")
        possible_paths = [
            Path(self.hf_home) / "hub" / f"models--{cache_id}" / "snapshots",
            Path(self.hf_home) / self.model_id,
        ]

        for snapshots_dir in possible_paths:
            if snapshots_dir.exists():
                snapshots = (
                    [d for d in snapshots_dir.iterdir() if d.is_dir()]
                    if snapshots_dir.is_dir()
                    else [snapshots_dir]
                )
                for snapshot in snapshots:
                    config_path = snapshot / "config.json"
                    if config_path.exists():
                        try:
                            with open(config_path) as f:
                                self.model_config = json.load(f)
                            self.log(f"Loaded model config from: {config_path}")
                            return
                        except Exception as e:
                            self.log(f"Error reading config: {e}", "error")

    def _init_calculator(self):
        """Initialize theoretical calculator."""
        # Detect hardware type
        if "H200" in self.gpu_name or self.gpu_memory_gb > 100:
            self.hardware = HardwareConfig.h200_cluster(self.gpu_count)
        elif "H100" in self.gpu_name:
            self.hardware = HardwareConfig.h100_cluster(self.gpu_count)
        elif "A100" in self.gpu_name:
            self.hardware = HardwareConfig.a100_80gb_cluster(self.gpu_count)
        else:
            self.hardware = HardwareConfig(
                gpu_count=self.gpu_count,
                gpu_memory_gb=self.gpu_memory_gb,
                gpu_name=self.gpu_name,
            )

        self.calculator = TheoreticalCalculator(self.hardware)
        self.log(f"Initialized calculator for {self.gpu_name}")

    def _extract_model_properties(self):
        """Extract model properties from configs."""
        # Get max_position_embeddings
        if self.model_config:
            tc = self.model_config.get("text_config", self.model_config)
            self.max_position_embeddings = tc.get(
                "max_position_embeddings", tc.get("n_positions", 32768)
            )
            self.log(f"Model max_position_embeddings: {self.max_position_embeddings:,}")

        # Detect capabilities from model ID
        capabilities = detect_model_capabilities(self.model_id)
        self.is_moe = capabilities["is_moe"]
        self.is_reasoning = capabilities["is_reasoning"]
        self.supports_tools = capabilities["supports_tools"]

        # Load vLLM extracted config first (most accurate)
        self._load_vllm_extracted_config()

        # Get model size - prefer vLLM extracted config over estimation
        if (
            self.vllm_extracted
            and self.vllm_extracted.success
            and self.vllm_extracted.total_params_b > 0
        ):
            self.model_params_b = self.vllm_extracted.total_params_b
            if self.vllm_extracted.is_moe and self.vllm_extracted.active_params_b > 0:
                self.log(
                    f"Model size: {self.model_params_b:.1f}B total parameters ({self.vllm_extracted.active_params_b:.1f}B active per token)"
                )
            else:
                self.log(f"Model size: {self.model_params_b:.1f}B parameters")
        else:
            # Fallback to estimation from checkpoint
            cache_path = get_model_cache_path(self.model_id, self.hf_home)
            self.model_params_b = get_model_size_from_id(self.model_id, cache_path)

            if self.model_params_b > 0:
                self.log(
                    f"Model size: {self.model_params_b:.1f}B parameters (estimated)"
                )
            else:
                self.log("Model size: Unknown (will use empirical testing)")

    def _load_vllm_extracted_config(self):
        """Load vLLM extracted configuration from model directory."""
        self.vllm_extracted = load_vllm_extracted_config(self.model_dir)

        if self.vllm_extracted and self.vllm_extracted.success:
            self.log(
                f"Loaded vLLM extracted config (v{self.vllm_extracted.vllm_version})"
            )
            self.log(f"  Architecture: {self.vllm_extracted.architecture}")
            self.log(f"  Model type: {self.vllm_extracted.model_type}")
            self.log(f"  Is MoE: {self.vllm_extracted.is_moe}")
            self.log(f"  Recommended TP: {self.vllm_extracted.recommended_tp}")
            if self.vllm_extracted.memory_estimate_gb > 0:
                self.log(
                    f"  Est. memory: {self.vllm_extracted.memory_estimate_gb:.1f}GB"
                )

            # Update is_moe from extracted config if available
            if self.vllm_extracted.is_moe:
                self.is_moe = True
        else:
            self.vllm_extracted = None
            self.log(
                "No vLLM extracted config found (run extract_vllm_config.py first)"
            )

    def _get_cuda_devices(self, tp: int) -> str:
        """Get CUDA_VISIBLE_DEVICES for given TP."""
        devices = self.cuda_devices.split(",")
        return ",".join(devices[:tp])

    def _get_venv_python(self) -> Path:
        venv_python = self.model_dir / ".venv" / "bin" / "python"
        if venv_python.exists():
            return venv_python
        return Path("/data/siraj/lm_start/.venv/bin/python")

    def _kill_process(self, process):
        """Kill a process group safely."""
        if process and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
            except:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass

    def _create_run_dir(self, context: int, attempt: int) -> Path:
        """Create a run directory for this test."""
        run_name = f"run_ctx{context}_attempt{attempt:02d}"
        run_dir = self.runs_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _build_vllm_command(self, config: Dict[str, Any], port: int) -> List[str]:
        flag_config = {
            "max_model_len": config["max_model_len"],
            "tensor_parallel_size": config["tensor_parallel_size"],
            "gpu_memory_utilization": config["gpu_memory_utilization"],
            "max_num_batched_tokens": config["max_num_batched_tokens"],
            "max_num_seqs": config["max_num_seqs"],
            "dtype": config.get("dtype", "auto"),
            "trust_remote_code": True,
        }

        if config.get("enable_chunked_prefill", True):
            flag_config["enable_chunked_prefill"] = True

        if config.get("kv_cache_dtype") and config["kv_cache_dtype"] != "auto":
            flag_config["kv_cache_dtype"] = config["kv_cache_dtype"]

        if self.is_moe or config.get("enable_expert_parallel", False):
            flag_config["enable_expert_parallel"] = True

        if config.get("enable_prefix_caching", True):
            flag_config["enable_prefix_caching"] = True

        if "max_cudagraph_capture_size" in config:
            flag_config["max_cudagraph_capture_size"] = config[
                "max_cudagraph_capture_size"
            ]

        if config.get("compilation_config"):
            flag_config["compilation_config"] = config["compilation_config"]

        if config.get("reasoning_parser"):
            flag_config["reasoning_parser"] = config["reasoning_parser"]

        if config.get("tool_call_parser"):
            flag_config["tool_call_parser"] = config["tool_call_parser"]
            flag_config["enable_auto_tool_choice"] = True

        if config.get("chat_template"):
            flag_config["chat_template"] = config["chat_template"]

        validated_args = validate_vllm_flags(flag_config, model_dir=str(self.model_dir))

        return [
            str(self._get_venv_python()),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_id,
            "--port",
            str(port),
        ] + validated_args

    def _test_config(self, config: Dict[str, Any], attempt: int) -> Dict:
        """Test a single vLLM configuration."""
        context = config["max_model_len"]
        run_dir = self._create_run_dir(context, attempt)

        port = 8000 + (attempt % 1000)
        tp = config["tensor_parallel_size"]

        cuda_devices = self._get_cuda_devices(tp)

        self.log(
            f"Testing: context={context:,}, TP={tp}, "
            f"batch_tokens={config['max_num_batched_tokens']}, "
            f"batch_seqs={config['max_num_seqs']}, "
            f"devices={cuda_devices}"
        )

        # Setup environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        env["VLLM_ATTENTION_BACKEND"] = config.get("attention_backend", "FLASHINFER")
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        env["HF_HOME"] = self.hf_home
        env["CUDA_HOME"] = self.cuda_home

        # Build command
        cmd = self._build_vllm_command(config, port)

        # Save config
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        result = {
            "success": False,
            "config": config,
            "run_dir": str(run_dir),
            "error": None,
            "error_type": None,
            "timestamp": datetime.now().isoformat(),
            "startup_time": None,
            "process": None,  # Will store process for cleanup after benchmark
        }

        process = None
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"

        try:
            with open(stdout_path, "w") as stdout_f, open(stderr_path, "w") as stderr_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    env=env,
                    preexec_fn=os.setsid,
                )

            start_time = time.time()

            # Wait for server to be ready
            while time.time() - start_time < self.VLLM_STARTUP_TIMEOUT:
                try:
                    resp = requests.get(f"http://localhost:{port}/health", timeout=5)
                    if resp.status_code == 200:
                        result["success"] = True
                        result["startup_time"] = time.time() - start_time
                        self.log(
                            f"✓ Server ready in {result['startup_time']:.1f}s",
                            "success",
                        )
                        break
                except:
                    pass

                if process.poll() is not None:
                    result["error"] = f"Process exited with code {process.returncode}"
                    break

                time.sleep(self.HEALTH_CHECK_INTERVAL)
            else:
                result["error"] = "Startup timeout"

        except Exception as e:
            result["error"] = str(e)

        # Store process in result for later cleanup (after benchmark)
        if result["success"]:
            result["process"] = process
        else:
            # Kill failed process immediately
            self._kill_process(process)

        # Analyze error
        if not result["success"] and stderr_path.exists():
            stderr = stderr_path.read_text().lower()
            if "out of memory" in stderr or "oom" in stderr or "cudaerror" in stderr:
                result["error_type"] = "cuda_oom"
            elif "gguf" in stderr:
                result["error_type"] = "incompatible_format"
            elif "module" in stderr and "not found" in stderr:
                result["error_type"] = "missing_module"
            elif "trust_remote_code" in stderr:
                result["error_type"] = "remote_code"
            elif "max_position_embeddings" in stderr or "context" in stderr:
                result["error_type"] = "context_too_large"
            else:
                result["error_type"] = "unknown"

        # Save result
        with open(run_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        if result["success"]:
            self.working_configs.append(config)
        else:
            self.failed_configs.append(
                {
                    "config": config,
                    "error": result["error"],
                    "error_type": result["error_type"],
                }
            )

        return result

    def _run_benchmark(self, config: Dict[str, Any], port: int) -> Optional[Dict]:
        """Run benchmark on a working configuration."""
        if not self.enable_benchmarking:
            return None

        self.log(f"Running benchmark for context={config['max_model_len']}")

        runner = BenchmarkRunner(
            model_id=self.model_id,
            vllm_port=port,
            venv_python=self._get_venv_python(),
            hf_home=self.hf_home,
            verbose=self.verbose,
        )

        try:
            result = runner.run_comprehensive_benchmark(
                config, quick_mode=self.quick_mode
            )
            self.benchmark_results.append(result)
            return result
        except Exception as e:
            self.log(f"Benchmark failed: {e}", "error")
            return None

    def _generate_initial_config(self) -> Dict[str, Any]:
        """Generate initial configuration from theoretical calculator."""
        # Create model config for calculator
        model = ModelConfig(
            model_id=self.model_id,
            model_params_b=self.model_params_b,
            max_position_embeddings=self.max_position_embeddings,
            num_layers=self.model_config.get("num_hidden_layers", 32)
            if self.model_config
            else 32,
            hidden_size=self.model_config.get("hidden_size", 4096)
            if self.model_config
            else 4096,
            num_attention_heads=self.model_config.get("num_attention_heads", 32)
            if self.model_config
            else 32,
            num_key_value_heads=self.model_config.get("num_key_value_heads", 8)
            if self.model_config
            else 8,
            is_moe=self.is_moe,
            is_reasoning=self.is_reasoning,
        )

        # Use vLLM extracted config if available
        if self.vllm_extracted and self.vllm_extracted.success:
            self.log("Using vLLM extracted configuration as base")

            # Determine TP size: use extracted recommendation
            # vLLM extraction calculates this based on actual memory requirements
            tp_size = self.vllm_extracted.recommended_tp
            if tp_size >= 1:
                self.log(f"  Using extracted TP recommendation: {tp_size}")
            else:
                # Fallback: use all GPUs (should not happen with proper extraction)
                tp_size = self.gpu_count
                self.log(f"  Fallback: Using all GPUs: TP={tp_size}")

            config = {
                "tensor_parallel_size": tp_size,
                "max_model_len": self.max_position_embeddings,
                "gpu_memory_utilization": 0.95,
                "max_num_batched_tokens": min(65536, self.max_position_embeddings * 2),
                "max_num_seqs": 256,
                "dtype": self.vllm_extracted.dtype or "auto",
                "attention_backend": "FLASHINFER",  # Will try extracted later if different
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "enable_expert_parallel": self.is_moe or self.vllm_extracted.is_moe,
            }

            # Store extracted values for potential use in optimization
            config["_vllm_extracted_backend"] = "FLASHINFER"  # vLLM auto-selected this
            config["_vllm_memory_estimate"] = self.vllm_extracted.memory_estimate_gb

        else:
            # Fall back to default configuration
            self.log("Using default configuration (no vLLM extraction)")
            config = {
                "tensor_parallel_size": self.gpu_count,  # USE ALL GPUS
                "max_model_len": self.max_position_embeddings,
                "gpu_memory_utilization": 0.95,
                "max_num_batched_tokens": min(65536, self.max_position_embeddings * 2),
                "max_num_seqs": 256,
                "dtype": "auto",
                "attention_backend": "FLASHINFER",
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "enable_expert_parallel": self.is_moe,
            }

        # Add parsers
        if self.is_reasoning:
            if "deepseek-r1" in self.model_id.lower():
                config["reasoning_parser"] = "deepseek_r1"
            elif "qwq" in self.model_id.lower():
                config["reasoning_parser"] = "qwq"
            elif "kimi" in self.model_id.lower():
                config["reasoning_parser"] = "kimi_k2"

        if self.supports_tools:
            if (
                "llama-3.2" in self.model_id.lower()
                or "llama3.2" in self.model_id.lower()
            ):
                config["tool_call_parser"] = "pythonic"
            elif (
                "llama-3" in self.model_id.lower() or "llama3" in self.model_id.lower()
            ):
                config["tool_call_parser"] = "llama3_json"
            elif "qwen" in self.model_id.lower():
                config["tool_call_parser"] = "hermes"
            elif (
                "mistral" in self.model_id.lower() or "mixtral" in self.model_id.lower()
            ):
                config["tool_call_parser"] = "mistral"

        # H200 optimizations
        if "H200" in self.gpu_name or self.gpu_memory_gb > 100:
            config["kv_cache_dtype"] = "fp8"
            config["max_cudagraph_capture_size"] = 128
            config["compilation_config"] = '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'

        return config

    def _generate_next_config(
        self, base_config: Dict[str, Any], reduction_phase: int, reduction_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Generate next configuration by reducing parameters.

        Args:
            base_config: Base configuration
            reduction_phase: Which parameter to reduce (index into reduction_priority)
            reduction_index: Which value to use from the parameter's values list

        Returns:
            New configuration or None if exhausted
        """
        if reduction_phase >= len(self.reduction_priority):
            return None

        param = self.reduction_priority[reduction_phase]
        param_name = param["name"]

        if reduction_index >= len(param["values"]):
            # Move to next phase
            return self._generate_next_config(base_config, reduction_phase + 1, 0)

        new_config = base_config.copy()
        new_config[param_name] = param["values"][reduction_index]

        return new_config

    def _reduce_context(self, current_context: int) -> int:
        """Reduce context length while maintaining reasonable targets."""
        reduction_levels = [
            int(self.max_position_embeddings * 0.75),
            int(self.max_position_embeddings * 0.50),
            int(self.max_position_embeddings * 0.25),
            32768,
            16384,
            8192,
            4096,
        ]

        for level in reduction_levels:
            if level < current_context:
                return level

        return max(4096, current_context // 2)

    def _intelligent_oom_recovery(
        self,
        current_config: Dict[str, Any],
        error_message: str,
        attempt_number: int,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Use VLLMFlagSelector to intelligently recover from OOM.

        This is the "master of inference" logic that dynamically discovers
        available flags and selects the best OOM mitigation strategy.
        """
        try:
            # Initialize flag selector with dynamically extracted flags
            selector = VLLMFlagSelector(self.model_dir)

            # Get intelligent OOM mitigation strategy
            next_config, reasoning = selector.suggest_next_config(
                current_config=current_config,
                error_type="cuda_oom",
                error_message=error_message,
                attempt_number=attempt_number,
            )

            return next_config, reasoning

        except FileNotFoundError:
            # vllm_available_flags.json not found - need to extract first
            self.log("Dynamic flags not available, falling back to basic reduction")
            return None, "Dynamic flag extraction not available"
        except Exception as e:
            self.log(f"Flag selector error: {e}")
            return None, f"Flag selector error: {e}"

    def run(self, context: Dict[str, Any]) -> AgentResult:
        """
        Run max-context-first optimization.

        Strategy:
        1. Start with max_position_embeddings
        2. Try aggressive batching
        3. Reduce batch parameters one by one
        4. Only reduce context if all batch parameters exhausted
        5. Benchmark every working config
        6. Return best config
        """
        self.status = AgentStatus.RUNNING

        self.log("=" * 70)
        self.log("MAX-CONTEXT-FIRST OPTIMIZATION")
        self.log("=" * 70)

        # Load all context
        self._load_all_context()

        # Check compatibility
        if not self.model_info.get("vllm_compatible", True):
            return AgentResult(
                success=False, message="Model incompatible with vLLM", actions_taken=[]
            )

        self.log(f"\nModel: {self.model_id}")
        self.log(f"Target Context: {self.max_position_embeddings:,}")
        self.log(f"Parameters: {self.model_params_b}B")
        self.log(f"Hardware: {self.gpu_count}x {self.gpu_name}")
        self.log(
            f"MoE: {self.is_moe}, Reasoning: {self.is_reasoning}, Tools: {self.supports_tools}"
        )

        # Generate initial config
        current_config = self._generate_initial_config()
        self.current_context_target = self.max_position_embeddings

        attempt = 0
        max_attempts = self.max_retries

        # Reduction state
        reduction_phase = 0
        reduction_index = 0

        while attempt < max_attempts:
            attempt += 1
            self.log(f"\n--- Attempt {attempt}/{max_attempts} ---")

            # Test current config
            result = self._test_config(current_config, attempt)

            if result["success"]:
                self.log(
                    f"✓ Working config found! Context: {current_config['max_model_len']:,}",
                    "success",
                )

                # Run benchmark (server is still running)
                if self.enable_benchmarking:
                    benchmark = self._run_benchmark(
                        current_config, 8000 + (attempt % 1000)
                    )
                    if benchmark:
                        result["benchmark"] = benchmark

                # Save to knowledge base
                self._save_to_kb(current_config, result)

                # Kill server process after benchmark is done
                if "process" in result and result["process"]:
                    self._kill_process(result["process"])

                # If we achieved max context, we're done
                if current_config["max_model_len"] >= self.max_position_embeddings:
                    self.log(
                        f"✓ Achieved maximum context! Optimization complete.", "success"
                    )
                    break

                # Try to optimize further - explore next parameter variations
                # Mark this config as baseline and try next variation
                next_config = self._generate_next_config(
                    current_config, reduction_phase, reduction_index
                )

                if next_config is not None:
                    self.log(f"Trying next parameter variation...")
                    current_config = next_config
                    reduction_index += 1

                    # If we've exhausted this phase's values, move to next phase
                    if reduction_index >= len(
                        self.reduction_priority[reduction_phase]["values"]
                    ):
                        reduction_phase += 1
                        reduction_index = 0
                else:
                    # Exhausted parameter variations, consider this config final
                    self.log(
                        f"✓ Config optimized. Context: {current_config['max_model_len']:,}",
                        "success",
                    )
                    break

            else:
                error_type = result.get("error_type", "unknown")
                self.log(f"✗ Failed: {error_type}", "error")

                # Use intelligent flag selector for OOM errors
                if error_type == "cuda_oom":
                    next_config, reasoning = self._intelligent_oom_recovery(
                        current_config, result.get("error", ""), attempt
                    )

                    if next_config:
                        self.log(f"OOM Recovery: {reasoning}")
                        current_config = next_config
                    else:
                        # Exhausted OOM strategies, reduce context
                        new_context = self._reduce_context(
                            current_config["max_model_len"]
                        )
                        if new_context < 4096:
                            self.log("Context reduced below 4K, giving up", "error")
                            break

                        self.log(
                            f"Reducing context from {current_config['max_model_len']:,} to {new_context:,}"
                        )
                        current_config = self._generate_initial_config()
                        current_config["max_model_len"] = new_context
                        self.current_context_target = new_context
                else:
                    # Use original reduction strategy for non-OOM errors
                    next_config = self._generate_next_config(
                        current_config, reduction_phase, reduction_index
                    )

                    if next_config is None:
                        # Exhausted all parameter reductions, need to reduce context
                        new_context = self._reduce_context(
                            current_config["max_model_len"]
                        )

                        if new_context < 4096:
                            self.log("Context reduced below 4K, giving up", "error")
                            break

                        self.log(
                            f"Reducing context from {current_config['max_model_len']:,} to {new_context:,}"
                        )
                        current_config = self._generate_initial_config()
                        current_config["max_model_len"] = new_context
                        self.current_context_target = new_context

                        # Reset reduction state
                        reduction_phase = 0
                        reduction_index = 0
                    else:
                        current_config = next_config
                        reduction_index += 1

                        # If we've exhausted this phase's values, move to next
                        if reduction_index >= len(
                            self.reduction_priority[reduction_phase]["values"]
                        ):
                            reduction_phase += 1
                            reduction_index = 0

        # Summary
        self.log("\n" + "=" * 70)
        self.log("OPTIMIZATION COMPLETE")
        self.log("=" * 70)

        if not self.working_configs:
            return AgentResult(
                success=False,
                message="No working configuration found",
                actions_taken=[f"tested_{attempt}"],
            )

        # Find best base config (highest context, then best benchmark)
        best_base_config = self._select_best_config()

        self.log(f"Working configs: {len(self.working_configs)}")
        self.log(f"Best base config:")
        self.log(f"  Context: {best_base_config.get('max_model_len', 0):,}")
        self.log(f"  TP: {best_base_config.get('tensor_parallel_size', 0)}")
        self.log(f"  Batch tokens: {best_base_config.get('max_num_batched_tokens', 0)}")
        self.log(f"  Batch seqs: {best_base_config.get('max_num_seqs', 0)}")

        # PHASE 2: Generate and test 3 profile variants
        profile_results = self._test_profile_variants(best_base_config)

        # Save all configs and comparison
        self._save_profile_configs(profile_results)

        # Generate final report
        self._generate_comparison_report(profile_results)

        return AgentResult(
            success=True,
            message=f"Optimization complete. Context={best_base_config.get('max_model_len', 0):,}, tested 3 profiles",
            actions_taken=[
                f"tested_{attempt}",
                f"working_{len(self.working_configs)}",
                "tested_3_profiles",
            ],
            metadata={
                "best_base_config": best_base_config,
                "profile_results": profile_results,
                "working_configs": self.working_configs,
                "benchmark_results": self.benchmark_results,
            },
        )

    def _select_best_config(self) -> Dict[str, Any]:
        """Select best configuration from working configs."""
        if not self.working_configs:
            return {}

        # Sort by: context (descending), then benchmark score
        def config_score(config):
            context = config.get("max_model_len", 0)

            # Find benchmark for this config
            benchmark_score = 0
            for br in self.benchmark_results:
                if br.get("config", {}).get("max_model_len") == context:
                    benchmark_score = br.get("summary", {}).get("overall_score", 0)
                    break

            # Weight: 60% context, 40% performance
            return context * 0.6 + benchmark_score * 0.4

        sorted_configs = sorted(self.working_configs, key=config_score, reverse=True)
        return sorted_configs[0]

    def _test_profile_variants(
        self, base_config: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate and test 3 profile variants (performance, quality, balanced).

        Returns dict with results for each profile.
        """
        self.log("\n" + "=" * 70)
        self.log("TESTING PROFILE VARIANTS")
        self.log("=" * 70)

        # Generate profile configs
        generator = ProfileConfigGenerator(self.model_dir, self.gpu_name)
        profiles = generator.generate_profiles(base_config)

        profile_results = {}

        for profile_name, profile_config in profiles.items():
            self.log(f"\n--- Testing {profile_name.upper()} profile ---")

            result = {
                "config": profile_config,
                "success": False,
                "benchmark": None,
                "error": None,
            }

            # Test the profile config
            test_result = self._test_config(
                profile_config, attempt=200 + len(profile_results)
            )

            if test_result["success"]:
                self.log(f"✓ {profile_name} profile working", "success")
                result["success"] = True

                # Run benchmark
                port = 8000 + (len(self.benchmark_results) % 1000)
                benchmark = self._run_benchmark(profile_config, port)

                if benchmark:
                    score = benchmark.get("summary", {}).get("overall_score", 0)
                    self.log(f"  Benchmark score: {score:.1f}")
                    result["benchmark"] = benchmark
                    self.benchmark_results.append(benchmark)

                # Save to knowledge base
                self._save_to_kb(profile_config, test_result)

                # Kill server
                if "process" in test_result and test_result["process"]:
                    self._kill_process(test_result["process"])
            else:
                error = test_result.get("error_type", "unknown")
                self.log(f"✗ {profile_name} profile failed: {error}", "error")
                result["error"] = error

            profile_results[profile_name] = result

        return profile_results

    def _save_profile_configs(self, profile_results: Dict[str, Dict[str, Any]]):
        """Save all profile configs to files."""
        profiles_dir = self.model_dir / ".profiles"
        profiles_dir.mkdir(exist_ok=True)

        for profile_name, result in profile_results.items():
            config = result.get("config", {})

            # Save clean config (without internal metadata)
            clean_config = {k: v for k, v in config.items() if not k.startswith("_")}

            profile_path = profiles_dir / f"{profile_name}_config.json"
            with open(profile_path, "w") as f:
                json.dump(clean_config, f, indent=2)

            # Save result with benchmark
            result_path = profiles_dir / f"{profile_name}_result.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

        self.log(f"\nProfile configs saved to: {profiles_dir}")

    def _generate_comparison_report(self, profile_results: Dict[str, Dict[str, Any]]):
        """Generate a comparison report of all profiles."""
        self.log("\n" + "=" * 70)
        self.log("PROFILE COMPARISON REPORT")
        self.log("=" * 70)

        # Build comparison table
        rows = []
        for profile_name, result in profile_results.items():
            config = result.get("config", {})
            benchmark = result.get("benchmark", {})
            summary = benchmark.get("summary", {}) if benchmark else {}

            rows.append(
                {
                    "profile": profile_name.upper(),
                    "status": "✓" if result.get("success") else "✗",
                    "context": config.get("max_model_len", 0),
                    "batch_tokens": config.get("max_num_batched_tokens", 0),
                    "batch_seqs": config.get("max_num_seqs", 0),
                    "kv_cache": config.get("kv_cache_dtype", "auto"),
                    "score": summary.get("overall_score", 0),
                    "ttft_ms": summary.get("avg_ttft_ms", 0),
                    "tput": summary.get("avg_throughput", 0),
                }
            )

        # Print table
        self.log(
            f"\n{'Profile':<12} {'Status':<8} {'Context':<10} {'Score':<8} {'TTFT':<10} {'Throughput':<12}"
        )
        self.log("-" * 70)
        for row in rows:
            self.log(
                f"{row['profile']:<12} {row['status']:<8} {row['context']:<10,} "
                f"{row['score']:<8.1f} {row['ttft_ms']:<10.1f} {row['tput']:<12.1f}"
            )

        # Save report
        report = {
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat(),
            "profiles": profile_results,
            "summary": {row["profile"].lower(): row for row in rows},
        }

        report_path = self.model_dir / "profile_comparison_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.log(f"\nComparison report saved to: {report_path}")

    def _save_to_kb(self, config: Dict, result: Dict):
        """Save working config to knowledge base."""
        if "verified_combinations" not in self.kb:
            self.kb["verified_combinations"] = []

        entry = {
            "tensor_parallel_size": config["tensor_parallel_size"],
            "max_model_len": config["max_model_len"],
            "max_num_batched_tokens": config.get("max_num_batched_tokens"),
            "max_num_seqs": config.get("max_num_seqs"),
            "gpu_memory_utilization": config["gpu_memory_utilization"],
            "attention_backend": config.get("attention_backend", "FLASHINFER"),
            "enable_chunked_prefill": config.get("enable_chunked_prefill", True),
            "startup_time": result.get("startup_time"),
            "timestamp": datetime.now().isoformat(),
            "run_dir": result.get("run_dir"),
            "benchmark": result.get("benchmark"),
        }

        self.kb["verified_combinations"].append(entry)
        self.kb["best_config"] = entry
        self.kb["_last_updated"] = datetime.now().isoformat()

        with open(self.kb_path, "w") as f:
            json.dump(self.kb, f, indent=2)

    def _save_final_config(self, config: Dict[str, Any]):
        """Save the final best configuration."""
        # Save to vllm_config.json (new format)
        final_config_path = self.model_dir / "vllm_config.json"

        final_config = {
            "model_id": self.model_id,
            "optimization_strategy": "max_context_first",
            "config": config,
            "hardware": {
                "gpu_count": self.gpu_count,
                "gpu_name": self.gpu_name,
                "gpu_memory_gb": self.gpu_memory_gb,
            },
            "model_properties": {
                "parameters_b": self.model_params_b,
                "max_position_embeddings": self.max_position_embeddings,
                "is_moe": self.is_moe,
                "is_reasoning": self.is_reasoning,
                "supports_tools": self.supports_tools,
            },
            "timestamp": datetime.now().isoformat(),
        }

        with open(final_config_path, "w") as f:
            json.dump(final_config, f, indent=2)

        self.log(f"Final config saved to: {final_config_path}")

        # Also save to optimized_config.json (legacy format for compatibility)
        optimized_config_path = self.model_dir / "optimized_config.json"

        # Build vLLM args list
        vllm_args = [
            "serve",
            self.model_id,
            "--port",
            str(config.get("port", 8000)),
            "--tensor-parallel-size",
            str(config.get("tensor_parallel_size", 8)),
            "--max-model-len",
            str(config.get("max_model_len", 4096)),
            "--gpu-memory-utilization",
            str(config.get("gpu_memory_utilization", 0.9)),
        ]

        # Add optional flags
        if config.get("dtype") and config.get("dtype") != "auto":
            vllm_args.extend(["--dtype", config["dtype"]])
        if config.get("enforce_eager"):
            vllm_args.append("--enforce-eager")
        if config.get("trust_remote_code") or self.model_info.get("trust_remote_code"):
            vllm_args.append("--trust-remote-code")
        if config.get("enable_chunked_prefill"):
            vllm_args.append("--enable-chunked-prefill")
        if config.get("enable_prefix_caching"):
            vllm_args.append("--enable-prefix-caching")

        # Find best benchmark result for this config
        best_context = config.get("max_model_len", 0)
        benchmark_score = 0
        for br in self.benchmark_results:
            if br.get("config", {}).get("max_model_len") == best_context:
                benchmark_score = br.get("summary", {}).get("overall_score", 0)
                break

        optimized_config = {
            "model_id": self.model_id,
            "vllm_args": vllm_args,
            "environment": {
                "CUDA_VISIBLE_DEVICES": self.cuda_devices,
                "HF_HOME": self.hf_home,
                "VLLM_ATTENTION_BACKEND": config.get("attention_backend", "FLASHINFER"),
            },
            "tensor_parallel_size": config.get("tensor_parallel_size", 8),
            "port": config.get("port", 8000),
            "host": "0.0.0.0",
            "max_model_len": best_context,
            "best_context": best_context,
            "is_satisfied": True,
            "benchmark_score": benchmark_score,
            "total_attempts": len(self.working_configs) + len(self.failed_configs),
            "timestamp": datetime.now().isoformat(),
        }

        with open(optimized_config_path, "w") as f:
            json.dump(optimized_config, f, indent=2)

        self.log(f"Optimized config saved to: {optimized_config_path}")
