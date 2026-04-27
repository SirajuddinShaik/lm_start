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
        # Bootstrap from .runs/ directory if history file doesn't exist yet
        return self._bootstrap_from_runs()

    def _bootstrap_from_runs(self) -> Dict:
        """Build history from existing .runs/ result.json files."""
        history = {
            "runs": [],
            "best_context": 0,
            "best_config": {},
            "attempted_targets": [],
            "satisfaction_score": 0.0,
            "is_satisfied": False,
            "total_attempts": 0,
        }
        runs_dir = self.history_file.parent / ".runs"
        if not runs_dir.exists():
            return history
        for run_dir in sorted(runs_dir.iterdir()):
            result_file = run_dir / "result.json"
            config_file = run_dir / "config.json"
            if not result_file.exists():
                continue
            try:
                result = json.loads(result_file.read_text())
                config = json.loads(config_file.read_text()) if config_file.exists() else {}
                success = result.get("success", False)
                max_len = config.get("max_model_len", 0)
                history["runs"].append({
                    "timestamp": result.get("timestamp_started", ""),
                    "target": max_len,
                    "success": success,
                    "config": config,
                    "error": result.get("error", ""),
                    "run_id": run_dir.name,
                })
                history["total_attempts"] += 1
                if success and max_len > history["best_context"]:
                    history["best_context"] = max_len
                    history["best_config"] = config.copy()
                if max_len and max_len not in history["attempted_targets"]:
                    history["attempted_targets"].append(max_len)
            except Exception:
                continue
        return history

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

    # Check if already optimized (informational only — never auto-stop)
    is_optimized, reason = history.is_already_optimized(min_context)
    if is_optimized:
        print(f"  Previously optimized ({reason}) — continuing to verify")
    else:
        print(f"  {reason}")

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
            f"  Previously optimized: reviewing {len(history.history['runs'])} previous run(s)..."
        )
        analysis = history.analyze_patterns()
        if analysis["recommendation"]:
            print(f"  Recommendation: {analysis['recommendation']}")

    start_time = time.time()

    if agentic:
        orchestrator = OpenCodeAgenticOrchestrator(
            model_dir=str(model_dir),
            model_id=model_id,
            goal=f"maximize context (minimum {min_context})",
            force=force,
        )
        orchestrator.run()
        # Reload history to pick up runs created during orchestration
        history = OptimizationHistory(model_dir_path)
        best_cfg = history.history.get("best_config") or base_config
        is_sat = history.history.get("is_satisfied", False)
        result["success"] = True
        result["config"] = best_cfg
        result["is_satisfied"] = is_sat
        return result
    else:
        agent = ExperimentAgentV2(
            model_dir=str(model_dir),
            max_retries=config.experiment.max_runs,
            verbose=verbose,
            hf_home=config.paths.hf_home,
        )
        agent_result = agent.run({})

    elapsed = time.time() - start_time

    # Record result (non-agentic path)
    best_config = agent_result.metadata.get("best_config", base_config)
    best_context = int(best_config.get("max_model_len", 0)) if best_config else 0
    improvements_made = agent_result.success or history.history["best_context"] > 0

    if best_config:
        history.add_run(target=best_context, success=agent_result.success, config=best_config, error="")

    if best_context >= min_context:
        history.history["is_satisfied"] = True
        history.save()

    result["success"] = agent_result.success
    result["config"] = best_config
    result["is_satisfied"] = history.history["is_satisfied"]
    return result


if __name__ == "__main__":
    sys.exit(main())
