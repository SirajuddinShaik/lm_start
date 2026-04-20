"""
Experiment Agent V2 - Intelligent Inference Optimizer

Unified agent that:
1. Calculates theoretical serving capacity from model + hardware
2. Builds optimal config based on calculations
3. Tests with adaptive retry (optimizations before reducing capacity)
4. Creates 3 variants (balanced, low-latency, high-throughput)
5. Tests all variants
6. Generates comparison report

Transparent reasoning at every step.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .base import BaseAgent, AgentResult, AgentStatus

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_start.utils.capacity_calculator import CapacityCalculator, CapacityEstimate
from lm_start.utils.intelligent_config import IntelligentConfigGenerator
from lm_start.utils.adaptive_validator import AdaptiveValidator, ValidationResult
from lm_start.utils.benchmark_runner import BenchmarkRunner


class ExperimentAgentV2(BaseAgent):
    """
    Intelligent agent for vLLM configuration optimization.

    Flow:
    1. CALCULATE → Capacity from model + hardware specs
    2. BUILD     → Optimal config from capacity estimate
    3. VALIDATE  → Test with adaptive retry
    4. BENCHMARK → Measure base config performance
    5. VARIANTS  → Create 3 profile variants
    6. COMPARE   → Test variants and generate report
    """

    def __init__(
        self,
        model_dir: str,
        max_retries: int = 10,
        verbose: bool = True,
        hf_home: Optional[str] = None,
    ):
        super().__init__("ExperimentAgentV2", model_dir, max_retries, verbose)
        self.model_dir = Path(model_dir)

        # Context
        self.model_info: Dict = {}
        self.device_config: Dict = {}
        self.model_config: Dict = {}

        # Hardware
        self.gpu_count = 8
        self.gpu_memory_gb = 141
        self.gpu_name = "unknown"
        self.cuda_devices = "0,1,2,3,4,5,6,7"
        self.hf_home = hf_home or os.environ.get("HF_HOME", "/data/.cache/huggingface")

        # Model
        self.model_id = ""
        self.max_position_embeddings = 32768
        self.model_params_b = 7.0
        self.is_moe = False
        self.is_reasoning = False

        # Results
        self.capacity_estimate: Optional[CapacityEstimate] = None
        self.base_config: Optional[Dict] = None
        self.base_validation: Optional[ValidationResult] = None
        self.base_benchmark: Optional[Dict] = None
        self.variant_results: Dict[str, Dict] = {}

        # Run tracking
        self.run_dir = self.model_dir / ".runs"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _load_context(self):
        """Load model and device configuration."""
        self.log("Loading context...")

        # Model info
        model_info_path = (
            self.model_dir / ".llm-context" / "model-context" / "model_info.json"
        )
        if model_info_path.exists():
            with open(model_info_path) as f:
                self.model_info = json.load(f)
            self.model_id = self.model_info.get("model_id", "")
            self.log(f"Model: {self.model_id}")

        # Device config
        device_config_path = (
            self.model_dir / ".llm-context" / "model-context" / "device_config.json"
        )
        if device_config_path.exists():
            with open(device_config_path) as f:
                self.device_config = json.load(f)
            gpus = self.device_config.get("gpus", {})
            self.gpu_count = gpus.get("count", 8)
            self.gpu_memory_gb = gpus.get("memory_gb_per_gpu", 141)
            self.gpu_name = gpus.get("name", "unknown")
            self.cuda_devices = gpus.get("visible_devices", "0,1,2,3,4,5,6,7")
            self.log(
                f"Hardware: {self.gpu_count}x {self.gpu_name} ({self.gpu_memory_gb}GB)"
            )

        # vLLM extracted config
        extracted_path = self.model_dir / "vllm_extracted_config.json"
        if extracted_path.exists():
            with open(extracted_path) as f:
                extracted = json.load(f)

            hf_config = extracted.get("hf_config", {})
            self.max_position_embeddings = hf_config.get(
                "max_position_embeddings", 32768
            )
            self.model_params_b = extracted.get("memory_estimate", {}).get(
                "parameters_b", 7.0
            )
            self.is_moe = extracted.get("is_moe", False)

            self.log(f"Max context: {self.max_position_embeddings:,}")
            self.log(f"Parameters: {self.model_params_b:.1f}B")
            self.log(f"MoE: {self.is_moe}")

    def run(self, context: Dict[str, Any]) -> AgentResult:
        """
        Run the full optimization pipeline.
        """
        self.status = AgentStatus.RUNNING

        self.log("=" * 70)
        self.log("INTELLIGENT INFERENCE OPTIMIZER V2")
        self.log("=" * 70)

        # Load context
        self._load_context()

        # PHASE 1: CALCULATE
        if not self._calculate_capacity():
            return AgentResult(
                success=False,
                message="Failed to calculate capacity",
                actions_taken=["load_context", "calculate_failed"],
            )

        # PHASE 2: BUILD & VALIDATE
        if not self._build_and_validate():
            return AgentResult(
                success=False,
                message="Failed to find working config",
                actions_taken=["load_context", "calculate", "validate_failed"],
            )

        # PHASE 3: BENCHMARK BASE
        self._benchmark_base()

        # PHASE 4: CREATE & TEST VARIANTS
        self._create_and_test_variants()

        # PHASE 5: GENERATE REPORT
        self._generate_final_report()

        base_seqs = self.base_config.get("max_num_seqs", 0) if self.base_config else 0
        base_len = self.base_config.get("max_model_len", 0) if self.base_config else 0

        return AgentResult(
            success=True,
            message=f"Optimization complete. Base config: {base_seqs} users @ {base_len:,} context",
            actions_taken=[
                "calculate_capacity",
                "build_validate",
                "benchmark_base",
                "test_variants",
                "generate_report",
            ],
            metadata={
                "capacity": self.capacity_estimate.to_dict()
                if self.capacity_estimate
                else {},
                "base_config": self.base_config or {},
                "base_benchmark": self.base_benchmark or {},
                "variants": self.variant_results,
            },
        )

    def _calculate_capacity(self) -> bool:
        """
        PHASE 1: Calculate theoretical serving capacity.
        """
        self.log("\n" + "=" * 70)
        self.log("PHASE 1: CALCULATE CAPACITY")
        self.log("=" * 70)

        calculator = CapacityCalculator()

        # Detect quantization from model name
        quantization = self._detect_quantization(self.model_id)

        # Get model specs from extracted config
        extracted_path = self.model_dir / "vllm_extracted_config.json"
        num_layers = 32
        hidden_size = 4096

        if extracted_path.exists():
            with open(extracted_path) as f:
                extracted = json.load(f)
            hf_config = extracted.get("hf_config", {})
            num_layers = hf_config.get("num_hidden_layers", 32)
            hidden_size = hf_config.get("hidden_size", 4096)

        self.log(
            f"Model specs: {self.model_params_b:.1f}B params, {num_layers} layers, {hidden_size} hidden"
        )
        self.log(f"Quantization: {quantization}")

        # Calculate capacity
        self.capacity_estimate = calculator.calculate(
            model_params=self.model_params_b * 1e9,
            num_layers=num_layers,
            hidden_size=hidden_size,
            max_context=self.max_position_embeddings,
            quantization=quantization,
            gpu_count=self.gpu_count,
            gpu_memory_gb=self.gpu_memory_gb,
            gpu_type=self.gpu_name,
            is_moe=self.is_moe,
        )

        # Print explanation
        for line in self.capacity_estimate.explain():
            self.log(line)

        return True

    def _build_and_validate(self) -> bool:
        """
        PHASE 2: Build config and validate with adaptive retry.
        """
        if self.capacity_estimate is None:
            self.log("No capacity estimate available", "error")
            return False

        self.log("\n" + "=" * 70)
        self.log("PHASE 2: BUILD & VALIDATE")
        self.log("=" * 70)

        # Generate base config from capacity
        generator = IntelligentConfigGenerator(self.model_dir)
        self.base_config = generator.generate(self.capacity_estimate)

        self.log("Generated base config:")
        self.log(f"  max_model_len: {self.base_config['max_model_len']:,}")
        self.log(f"  max_num_seqs: {self.base_config['max_num_seqs']}")
        self.log(
            f"  max_num_batched_tokens: {self.base_config['max_num_batched_tokens']:,}"
        )
        self.log(f"  kv_cache_dtype: {self.base_config['kv_cache_dtype']}")

        # Validate with adaptive retry
        validator = AdaptiveValidator(
            model_dir=self.model_dir,
            model_id=self.model_id,
            venv_python=self.model_dir / ".venv" / "bin" / "python",
            hf_home=self.hf_home,
            verbose=self.verbose,
        )

        result, final_config = validator.validate(
            self.base_config, max_retries=self.max_retries
        )

        if not result.success:
            self.log("Failed to find working configuration after all retries", "error")
            return False

        self.base_validation = result
        self.base_config = final_config  # May be modified by validator

        # Log the final working configuration with reasoning
        self.log(
            "[REASONING] Validation successful. Agent adapted config from initial estimate."
        )
        self.log("[REASONING] Changes made during validation:")

        # Show what changed
        if final_config.get("max_num_seqs") != self.base_config.get("max_num_seqs"):
            self.log(f"  - max_num_seqs: Adjusted to fit memory constraints")
        if final_config.get("max_model_len") != self.base_config.get("max_model_len"):
            original = self.base_config.get("max_model_len", 0)
            actual = final_config.get("max_model_len", 0)
            self.log(
                f"  - max_model_len: {original:,} → {actual:,} (OOM required reduction)"
            )

        self.log(f"✓ Working config found:")
        self.log(f"  max_model_len: {self.base_config['max_model_len']:,}")
        self.log(f"  max_num_seqs: {self.base_config['max_num_seqs']}")
        self.log(f"  Startup time: {result.startup_time:.1f}s")

        # Save working config
        self._save_config(self.base_config, "base", result)

        # Kill the server (we'll restart for benchmark)
        validator._kill_process(result.process)

        return True

    def _benchmark_base(self):
        """
        PHASE 3: Benchmark the base configuration.
        """
        self.log("\n" + "=" * 70)
        self.log("PHASE 3: BENCHMARK BASE CONFIG")
        self.log("=" * 70)

        # Re-start server for benchmark
        validator = AdaptiveValidator(
            model_dir=self.model_dir,
            model_id=self.model_id,
            venv_python=self.model_dir / ".venv" / "bin" / "python",
            hf_home=self.hf_home,
            verbose=False,  # Less noise during benchmark
        )

        result, _ = validator.validate(self.base_config, max_retries=1)

        if not result.success:
            self.log("Could not restart server for benchmark", "warning")
            return

        # Run benchmark
        runner = BenchmarkRunner(
            model_id=self.model_id,
            vllm_port=result.port,
            venv_python=self.model_dir / ".venv" / "bin" / "python",
            hf_home=self.hf_home,
            verbose=self.verbose,
        )

        self.base_benchmark = runner.run_comprehensive_benchmark(
            config=self.base_config,
            quick_mode=False,
        )

        if self.base_benchmark:
            summary = self.base_benchmark.get("summary", {})
            self.log(f"Benchmark complete:")
            self.log(f"  Score: {summary.get('overall_score', 0):.1f}/100")
            self.log(f"  Throughput: {summary.get('avg_throughput', 0):.1f} tok/s")
            self.log(f"  Avg TTFT: {summary.get('avg_ttft_ms', 0):.1f}ms")

        # Kill server
        validator._kill_process(result.process)

    def _create_and_test_variants(self):
        """
        PHASE 4: Create and test 3 variants.
        """
        self.log("\n" + "=" * 70)
        self.log("PHASE 4: CREATE & TEST VARIANTS")
        self.log("=" * 70)

        generator = IntelligentConfigGenerator(self.model_dir)
        variants = generator.create_variants(self.base_config, self.capacity_estimate)

        for name, config in variants.items():
            self.log(f"\n--- Testing {name.upper()} variant ---")

            validator = AdaptiveValidator(
                model_dir=self.model_dir,
                model_id=self.model_id,
                venv_python=self.model_dir / ".venv" / "bin" / "python",
                hf_home=self.hf_home,
                verbose=False,
            )

            result, final_config = validator.validate(config, max_retries=3)

            variant_result = {
                "config": final_config,
                "success": result.success,
                "error": result.error_type if not result.success else None,
                "benchmark": None,
            }

            if result.success:
                self.log(f"✓ {name} variant working")

                # Quick benchmark
                runner = BenchmarkRunner(
                    model_id=self.model_id,
                    vllm_port=result.port,
                    venv_python=self.model_dir / ".venv" / "bin" / "python",
                    hf_home=self.hf_home,
                    verbose=False,
                )

                # Shorter benchmark for variants (quick_mode=True)
                benchmark = runner.run_comprehensive_benchmark(
                    config=final_config,
                    quick_mode=True,
                )

                variant_result["benchmark"] = benchmark

                if benchmark:
                    summary = benchmark.get("summary", {})
                    self.log(f"  Score: {summary.get('overall_score', 0):.1f}")

                validator._kill_process(result.process)
            else:
                self.log(f"✗ {name} variant failed: {result.error_type}", "error")

            self.variant_results[name] = variant_result
            self._save_config(final_config, name, result)

    def _generate_final_report(self):
        """
        PHASE 5: Generate comparison report.
        """
        self.log("\n" + "=" * 70)
        self.log("PHASE 5: COMPARISON REPORT")
        self.log("=" * 70)

        # Build comparison table
        rows = []

        # Base config
        if self.base_benchmark:
            summary = self.base_benchmark.get("summary", {})
            rows.append(
                {
                    "profile": "BASE (balanced)",
                    "success": True,
                    "users": self.base_config.get("max_num_seqs", 0),
                    "context": self.base_config.get("max_model_len", 0),
                    "score": summary.get("overall_score", 0),
                    "throughput": summary.get("avg_throughput", 0),
                    "ttft": summary.get("avg_ttft_ms", 0),
                }
            )

        # Variants
        for name, result in self.variant_results.items():
            benchmark = result.get("benchmark", {})
            summary = benchmark.get("summary", {}) if benchmark else {}
            config = result.get("config", {})

            rows.append(
                {
                    "profile": name.upper(),
                    "success": result.get("success", False),
                    "users": config.get("max_num_seqs", 0),
                    "context": config.get("max_model_len", 0),
                    "score": summary.get("overall_score", 0),
                    "throughput": summary.get("avg_throughput", 0),
                    "ttft": summary.get("avg_ttft_ms", 0),
                }
            )

        # Print table
        self.log(
            f"\n{'Profile':<18} {'Status':<8} {'Users':<8} {'Context':<10} {'Score':<8} {'TTFT':<10} {'Tput':<12}"
        )
        self.log("-" * 90)
        for row in rows:
            status = "✓" if row["success"] else "✗"
            self.log(
                f"{row['profile']:<18} {status:<8} {row['users']:<8} "
                f"{row['context']:<10,} {row['score']:<8.1f} "
                f"{row['ttft']:<10.1f} {row['throughput']:<12.1f}"
            )

        # Determine best config
        successful = [r for r in rows if r["success"]]
        if successful:
            best = max(successful, key=lambda r: r["score"])
            self.log(f"\n★ BEST CONFIG: {best['profile']} (score: {best['score']:.1f})")

        # Save report
        report = {
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat(),
            "capacity": self.capacity_estimate.to_dict()
            if self.capacity_estimate
            else {},
            "base_config": self.base_config,
            "base_benchmark": self.base_benchmark,
            "variants": self.variant_results,
            "comparison": rows,
            "best_profile": best["profile"] if successful else None,
        }

        report_path = self.model_dir / "optimization_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.log(f"\nFull report saved to: {report_path}")

    def _save_config(self, config: Dict, name: str, result: ValidationResult):
        """Save config to runs directory."""
        run_subdir = self.run_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_subdir.mkdir(exist_ok=True)

        # Save config
        with open(run_subdir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save result
        with open(run_subdir / "result.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def _detect_quantization(self, model_id: str) -> str:
        """Detect quantization from model name."""
        model_lower = model_id.lower()
        if "w4a16" in model_lower or "awq" in model_lower or "gptq" in model_lower:
            return "w4a16"
        elif "w8a8" in model_lower or "fp8" in model_lower:
            return "w8a8"
        elif "bf16" in model_lower:
            return "bf16"
        return "fp16"


# Entry point
def run_optimization(model_dir: str, **kwargs) -> AgentResult:
    """Run full optimization pipeline."""
    agent = ExperimentAgentV2(model_dir, **kwargs)
    return agent.run({})
