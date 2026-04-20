"""
Phase Agent Module

Handles failures at the setup phase level (venv, download, config generation).
This is Agent Level 1 - handles infrastructure and setup issues.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import BaseAgent, AgentResult, AgentStatus, FixStrategy
from lm_start.utils.prompt_manager import get_prompt


class PhaseAgent(BaseAgent):
    """
    Agent for handling setup phase failures.

    This agent runs when a deterministic phase fails (init, fetch_info,
    download, venv, generate_config). It analyzes the error and attempts
    automatic recovery.
    """

    # Error patterns and their typical causes
    ERROR_PATTERNS = {
        "DOWNLOAD_FAILED": [
            "download failed",
            "connection error",
            "timeout",
            "403 forbidden",
            "authentication failed",
        ],
        "VENV_FAILED": [
            "failed to create virtual environment",
            "pip install failed",
            "no module named",
            "cuda not available",
        ],
        "CONFIG_FAILED": [
            "failed to generate",
            "template not found",
            "permission denied",
        ],
        "CUDA_ERROR": ["cuda out of memory", "cuda error", "no gpu found"],
        "DISK_ERROR": ["no space left", "permission denied", "read-only file system"],
    }

    def __init__(self, model_dir: str, max_retries: int = 5, verbose: bool = False):
        super().__init__("PhaseAgent", model_dir, max_retries, verbose)
        self._setup_strategies()

    def _setup_strategies(self) -> None:
        """Setup available fix strategies."""
        self.strategies = [
            FixStrategy(
                name="retry_download",
                description="Retry model download with different mirror",
                applies_to=["DOWNLOAD_FAILED"],
                action=self._retry_download,
            ),
            FixStrategy(
                name="fix_venv_pip",
                description="Fix pip installation in venv",
                applies_to=["VENV_FAILED"],
                action=self._fix_venv_pip,
            ),
            FixStrategy(
                name="fix_cuda_paths",
                description="Fix CUDA library paths",
                applies_to=["CUDA_ERROR", "VENV_FAILED"],
                action=self._fix_cuda_paths,
            ),
            FixStrategy(
                name="free_disk_space",
                description="Clean up disk space",
                applies_to=["DISK_ERROR"],
                action=self._free_disk_space,
            ),
            FixStrategy(
                name="regenerate_config",
                description="Regenerate configuration files",
                applies_to=["CONFIG_FAILED"],
                action=self._regenerate_config,
            ),
        ]

    def run(self, context: Dict[str, Any]) -> AgentResult:
        """
        Execute phase recovery.

        Args:
            context: Must contain:
                - phase: The failed phase name
                - error: Error message
                - error_type: Classified error type (optional)

        Returns:
            AgentResult with recovery actions
        """
        self.status = AgentStatus.RUNNING
        phase = context.get("phase", "unknown")
        error = context.get("error", "")
        error_type = context.get("error_type") or self._classify_error(error)

        self.log(f"Handling failure in phase: {phase}")
        self.log(f"Error type: {error_type}")
        self.log(f"Error: {error[:200]}")

        # Try automatic fixes first
        for attempt in range(1, self.max_retries + 1):
            self.attempt_count = attempt
            self.log(f"Attempt {attempt}/{self.max_retries}")

            # Find applicable strategy
            strategy = self._find_strategy(error_type)
            if strategy:
                self.log(f"Applying strategy: {strategy.name}")
                result = strategy.apply(context)
                self.record_action(
                    strategy.name, "attempted", {"success": result.success}
                )

                if result.success:
                    self.status = AgentStatus.SUCCESS
                    return AgentResult(
                        success=True,
                        message=f"Phase {phase} recovered using {strategy.name}",
                        actions_taken=[strategy.name],
                        metadata={"attempts": attempt},
                    )

            # If no strategy worked, consult LLM
            if attempt == self.max_retries:
                self.log("Automatic fixes failed, consulting LLM...")
                llm_result = self._consult_llm_for_fix(phase, error, error_type)
                if llm_result:
                    return llm_result

        self.status = AgentStatus.FAILED
        return AgentResult(
            success=False,
            message=f"Failed to recover phase {phase} after {self.max_retries} attempts",
            error_type=error_type,
            error_details=error,
            actions_taken=[a["action"] for a in self.action_history],
        )

    def _classify_error(self, error: str) -> str:
        """Classify error type from message."""
        error_lower = error.lower()

        for error_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in error_lower:
                    return error_type

        return "UNKNOWN"

    def _find_strategy(self, error_type: str) -> Optional[FixStrategy]:
        """Find a strategy that can handle this error type."""
        for strategy in self.strategies:
            if strategy.can_apply(error_type):
                return strategy
        return None

    def _retry_download(self, context: Dict[str, Any]) -> AgentResult:
        """Retry download with fallback options."""
        model_id = context.get("model_id", "")

        try:
            # Try with HF_HUB_OFFLINE=0 to force online
            env = os.environ.copy()
            env["HF_HUB_OFFLINE"] = "0"
            env["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

            # Run download script
            if self.model_dir:
                download_script = (
                    Path(__file__).parent.parent / "phases" / "download.py"
                )
                if download_script.exists():
                    result = subprocess.run(
                        [
                            "python3",
                            str(download_script),
                            "--model-dir",
                            str(self.model_dir),
                        ],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=600,
                    )

                    if result.returncode == 0:
                        return AgentResult(
                            success=True,
                            message="Download succeeded on retry",
                            actions_taken=["retry_download"],
                        )

            return AgentResult(
                success=False,
                message="Download retry failed",
                error_type="DOWNLOAD_FAILED",
            )

        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Download retry error: {e}",
                error_type="EXCEPTION",
                error_details=str(e),
            )

    def _fix_venv_pip(self, context: Dict[str, Any]) -> AgentResult:
        """Fix pip installation issues in venv."""
        if not self.model_dir:
            return AgentResult(success=False, message="No model directory")

        venv_path = self.model_dir / ".venv"

        try:
            # Ensure pip is installed
            python_bin = venv_path / "bin" / "python"
            if python_bin.exists():
                subprocess.run(
                    [str(python_bin), "-m", "ensurepip", "--upgrade"],
                    capture_output=True,
                    check=False,
                )

                # Upgrade pip
                subprocess.run(
                    [str(python_bin), "-m", "pip", "install", "--upgrade", "pip"],
                    capture_output=True,
                    check=False,
                )

                return AgentResult(
                    success=True,
                    message="Pip fixed in venv",
                    actions_taken=["fix_venv_pip"],
                )

            return AgentResult(success=False, message="Python binary not found in venv")

        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Failed to fix venv pip: {e}",
                error_type="EXCEPTION",
            )

    def _fix_cuda_paths(self, context: Dict[str, Any]) -> AgentResult:
        """Fix CUDA library paths."""
        try:
            # Detect CUDA paths
            from core.hardware import get_hardware_info

            hardware = get_hardware_info()

            if hardware.cuda_version:
                cuda_home = f"/usr/local/cuda-{hardware.cuda_version}"
                if Path(cuda_home).exists():
                    # Set environment variables
                    os.environ["CUDA_HOME"] = cuda_home
                    os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
                    os.environ["LD_LIBRARY_PATH"] = (
                        f"{cuda_home}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
                    )

                    return AgentResult(
                        success=True,
                        message=f"CUDA paths set to {cuda_home}",
                        actions_taken=["fix_cuda_paths"],
                    )

            return AgentResult(
                success=False, message="Could not detect CUDA installation"
            )

        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Failed to fix CUDA paths: {e}",
                error_type="EXCEPTION",
            )

    def _free_disk_space(self, context: Dict[str, Any]) -> AgentResult:
        """Attempt to free disk space."""
        try:
            actions = []

            # Clean pip cache
            subprocess.run(["pip", "cache", "purge"], capture_output=True, check=False)
            actions.append("pip_cache_purged")

            # Clean temp files
            import tempfile
            import shutil

            temp_dir = tempfile.gettempdir()
            for item in Path(temp_dir).glob("tmp*"):
                try:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                except:
                    pass
            actions.append("temp_cleaned")

            return AgentResult(
                success=True,
                message="Disk space cleanup completed",
                actions_taken=actions,
            )

        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Failed to free disk space: {e}",
                error_type="EXCEPTION",
            )

    def _regenerate_config(self, context: Dict[str, Any]) -> AgentResult:
        """Regenerate configuration files."""
        try:
            # Remove old configs
            if self.model_dir:
                for config_file in ["ecosystem.config.js", "model.sh"]:
                    config_path = self.model_dir / config_file
                    if config_path.exists():
                        config_path.unlink()

            return AgentResult(
                success=True,
                message="Configuration files marked for regeneration",
                actions_taken=["regenerate_config"],
            )

        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Failed to regenerate config: {e}",
                error_type="EXCEPTION",
            )

    def _consult_llm_for_fix(
        self, phase: str, error: str, error_type: str
    ) -> Optional[AgentResult]:
        """Consult LLM for a custom fix."""
        prompt = get_prompt(
            "phase_recovery.phase_failure_analysis",
            context={
                "phase": phase,
                "error_type": error_type,
                "error": error,
                "action_history": json.dumps(self.action_history, indent=2),
            },
        )

        response = self.consult_llm(prompt)

        try:
            fix_data = json.loads(response)
            fix = fix_data.get("fix", {})

            if fix.get("command"):
                # Execute the suggested command
                result = subprocess.run(
                    fix["command"],
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode == 0:
                    return AgentResult(
                        success=True,
                        message=f"LLM fix applied: {fix.get('description', '')}",
                        actions_taken=["llm_fix", fix.get("action", "")],
                        metadata={"llm_analysis": fix_data.get("analysis", "")},
                    )

            return AgentResult(
                success=False,
                message="LLM suggested fix failed",
                actions_taken=["llm_fix_attempted"],
            )

        except json.JSONDecodeError:
            return AgentResult(
                success=False,
                message="Could not parse LLM response",
                actions_taken=["llm_parse_failed"],
            )
