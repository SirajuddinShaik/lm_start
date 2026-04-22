#!/usr/bin/env python3
"""
Advanced Agentic vLLM Optimization

Iterative optimization that:
1. Reviews previous optimization history
2. Determines if model is already optimized 
3. Runs sequentially with intelligent retries
4. Analyzes all results and does additional runs if needed
5. Continues until satisfaction criteria met
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from lm_start.agents import ExperimentAgentV2
from lm_start.agents.base import AgentResult
from lm_start.core.config import get_config
from lm_start.scripts.orchestrator import OpenCodeAgenticOrchestrator
from lm_start.vllm_flag_validator import validate_vllm_flags


class OptimizationHistory:
    """Tracks optimization history across runs."""

    def __init__(self, model_dir: Path):
        self.history_file = model_dir / ".optimization_history.json"
        self.history = self._load()

    def _load(self) -> Dict:
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return {
            "runs": [],
            "best_context": 0,
            "best_config": {},
            "attempted_targets": [],
            "satisfaction_score": 0.0,
            "is_satisfied": False,
            "total_attempts": 0,
        }

    def save(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def add_run(self, target: int, success: bool, config: Dict, error: str = ""):
        run = {
            "timestamp": datetime.now().isoformat(),
            "target": target,
            "success": success,
            "config": config,
            "error": error,
        }
        self.history["runs"].append(run)
        self.history["total_attempts"] += 1

        if success and target > self.history["best_context"]:
            self.history["best_context"] = target
            self.history["best_config"] = config.copy()

        if target not in self.history["attempted_targets"]:
            self.history["attempted_targets"].append(target)

        self.save()

    def is_already_optimized(self, min_context: int = 8192) -> Tuple[bool, str]:
        """Check if model is already sufficiently optimized."""
        if not self.history["runs"]:
            return False, "No previous optimization runs found"

        best = self.history["best_context"]
        attempts = len(self.history["attempted_targets"])

        if best >= min_context:
            return True, f"Already optimized to {best} context (minimum {min_context})"

        if attempts >= 10 and best < 4096:
            return True, f"Multiple attempts ({attempts}) made, best is {best}"

        # Check if we tried all targets and failed
        if self.history["is_satisfied"]:
            return True, "Marked as satisfied in previous run"

        return False, f"Previous best: {best}, will try to improve"

    def get_next_target(self, context_targets: List[int]) -> Optional[int]:
        """Get next context target to try, considering history."""
        for target in context_targets:
            # Skip if already succeeded at this or higher
            if target <= self.history["best_context"]:
                continue

            # Skip if failed multiple times at this target
            failures = sum(
                1
                for r in self.history["runs"]
                if r["target"] == target and not r["success"]
            )
            if failures >= 3:
                continue

            return target

        return None

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze failure patterns to suggest better strategies."""
        failures = [r for r in self.history["runs"] if not r["success"]]
        successes = [r for r in self.history["runs"] if r["success"]]

        analysis = {"common_errors": {}, "successful_configs": [], "recommendation": ""}

        # Count error types
        for run in failures:
            error = run.get("error", "Unknown")
            analysis["common_errors"][error] = (
                analysis["common_errors"].get(error, 0) + 1
            )

        # Collect successful configs
        for run in successes:
            analysis["successful_configs"].append(
                {"target": run["target"], "config": run["config"]}
            )

        # Generate recommendation
        if "CUDA_OOM" in analysis["common_errors"]:
            analysis["recommendation"] = (
                "Try reducing memory utilization or enabling quantization"
            )
        elif "CONTEXT_TOO_LONG" in analysis["common_errors"]:
            analysis["recommendation"] = "Model architecture may limit context length"

        return analysis


def optimize_vllm_config(
    model_dir: str,
    min_context: int = 8192,
    max_iterations: int = 3,
    force: bool = False,
    verbose: bool = False,
    agentic: bool = True,
) -> Dict[str, Any]:
    """Optimize vLLM configuration for the model.

    Returns:
        Dict with 'success', 'config', 'is_satisfied', and 'error' keys
    """
    result = {
        "success": False,
        "config": None,
        "is_satisfied": False,
        "error": None,
    }

    model_dir_path = Path(model_dir)

    # Load model info
    model_info_file = (
        model_dir_path / ".llm-context" / "model-context" / "model_info.json"
    )
    if not model_info_file.exists():
        result["error"] = f"model_info.json not found at {model_info_file}"
        return result

    with open(model_info_file) as f:
        model_info = json.load(f)

    model_id = model_info.get("model_id", "")

    # Load optimization history
    history = OptimizationHistory(model_dir_path)

    # Check if already optimized
    is_optimized, reason = history.is_already_optimized(min_context)
    if is_optimized and not force:
        print(f"\n[INFO] Model appears to be already optimized.")
        print(f"       {reason}")
        print(f"\nUse force=True to re-optimize anyway.")
        result["success"] = True
        result["config"] = history.history.get("best_config", {})
        result["is_satisfied"] = True
        return result
    elif is_optimized and force:
        print(f"\n[INFO] Forcing re-optimization despite: {reason}")
    else:
        print(f"\n[INFO] {reason}")

    # Get config
    config = get_config()

    # Default base config (used if no iterations run)
    base_config = {
        "model": model_id,
        "tensor_parallel_size": config.device.gpu.count,
        "gpu_memory_utilization": config.vllm_defaults.gpu_memory_utilization,
        "dtype": config.vllm_defaults.dtype,
        "enforce_eager": config.vllm_defaults.enforce_eager,
        "trust_remote_code": config.vllm_defaults.trust_remote_code,
    }

    # Analyze previous patterns
    if history.history["runs"]:
        print(
            f"\n[ANALYSIS] Reviewing {len(history.history['runs'])} previous attempts..."
        )
        analysis = history.analyze_patterns()
        if analysis["recommendation"]:
            print(f"           Recommendation: {analysis['recommendation']}")

    start_time = time.time()

    opt_result = None
    if agentic:
        orchestrator = OpenCodeAgenticOrchestrator(
            model_dir=str(model_dir),
            model_id=model_id,
            goal=f"maximize context (minimum {min_context})",
            force=force,
        )
        orchestrator.run()
        result = AgentResult(
            success=True,
            message="Agentic optimization complete",
            metadata={
                "best_config": history.history.get("best_config", {}),
                "total_tests": 0,
            },
        )
    else:
        agent = ExperimentAgentV2(
            model_dir=str(model_dir),
            max_retries=config.experiment.max_runs,
            verbose=verbose,
            hf_home=config.paths.hf_home,
        )
        result = agent.run({})

    elapsed = time.time() - start_time

    # Record result
    best_config = result.metadata.get("best_config", base_config)
    best_throughput = result.metadata.get("best_throughput", 0)

    # Get context from max_model_len (not context_length)
    best_context = best_config.get("max_model_len", 0) if best_config else 0

    if best_config:
        history.add_run(
            target=best_context,
            success=result.success,
            config=best_config,
            error="",
        )
        improvements_made = result.success or history.history["best_context"] > 0

    if result.success:
        print(f"\n[SUCCESS] Optimization complete! ({elapsed:.1f}s)")
        print(f"          Tests run: {result.metadata.get('total_tests', 0)}")
        print(f"          Best throughput: {best_throughput} tokens/sec")
    else:
        print(f"\n[FAILED] Optimization did not improve ({elapsed:.1f}s)")

    if best_context >= min_context:
        history.history["is_satisfied"] = True
        history.save()

    # Final analysis
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}")

    print(f"\n[RESULTS]")
    print(f"  Total attempts: {history.history['total_attempts']}")
    print(f"  Best context: {history.history['best_context']}")
    print(f"  Targets tried: {len(history.history['attempted_targets'])}")

    if history.history["best_config"]:
        print(f"\n[BEST CONFIGURATION]")
        for key, value in history.history["best_config"].items():
            print(f"  {key}: {value}")

    output_file = model_dir / ".llm-context" / "model-context" / "optimized_config.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    best_config = history.history["best_config"] or base_config

    flag_config = {
        "tensor_parallel_size": best_config.get("tensor_parallel_size", 1),
        "max_model_len": history.history["best_context"] or 4096,
        "gpu_memory_utilization": best_config.get("gpu_memory_utilization", 0.9),
    }

    if best_config.get("dtype") and best_config.get("dtype") != "auto":
        flag_config["dtype"] = best_config["dtype"]
    if best_config.get("enforce_eager"):
        flag_config["enforce_eager"] = True
    if best_config.get("trust_remote_code"):
        flag_config["trust_remote_code"] = True

    validated_args = validate_vllm_flags(flag_config, model_dir=str(model_dir))
    vllm_args = ["serve", model_id] + validated_args

    with open(output_file, "w") as f:
        json.dump(
            {
                "model_id": model_id,
                "vllm_args": vllm_args,
                "environment": {
                    "CUDA_VISIBLE_DEVICES": config.device.gpu.visible_devices,
                    "HF_HOME": config.paths.hf_home,
                    "VLLM_ATTENTION_BACKEND": config.environment.get(
                        "VLLM_ATTENTION_BACKEND", "FLASHINFER"
                    ),
                },
                "tensor_parallel_size": best_config.get("tensor_parallel_size", 1),
                "port": config.vllm_defaults.port,
                "host": config.vllm_defaults.host,
                "max_model_len": history.history["best_context"] or 4096,
                "best_context": history.history["best_context"],
                "is_satisfied": history.history["is_satisfied"],
                "total_attempts": history.history["total_attempts"],
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"\n[OUTPUT] Config saved to: {output_file}")

    if history.history["is_satisfied"]:
        print(f"\n✓ SATISFIED: Achieved target context {min_context}")
        return 0
    elif improvements_made:
        print(f"\n⚠ PARTIAL: Made improvements but didn't reach {min_context}")
        return 0
    else:
        print(f"\n✗ NO IMPROVEMENT: Could not improve beyond current config")
        return 1


if __name__ == "__main__":
    sys.exit(main())
