"""
Orchestrator for coordinating model deployment phases.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from lm_start.agents.phase_agent import PhaseAgent
from lm_start.agents.experiment_agent import ExperimentAgent
from lm_start.agents.experiment_agent_v2 import ExperimentAgentV2
from lm_start.core.config import get_config
from lm_start.utils.logger import get_logger


class Orchestrator:
    """Main orchestrator for model deployment pipeline."""

    def __init__(self, model_dir: str, verbose: bool = False):
        self.model_dir = Path(model_dir)
        self.logger = get_logger("orchestrator", verbose)
        self.config = get_config()
        self.state_file = self.model_dir / "setup_state.json"

        # Define standard phases
        self.phases = ["download", "create_venv", "generate_config", "optimize"]

    def _load_state(self) -> Dict[str, Any]:
        """Load setup state from file."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {"completed_phases": [], "phases": {}}

    def _save_state(self, state: Dict[str, Any]) -> None:
        """Save setup state to file."""
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _is_phase_completed(self, phase: str, state: Dict[str, Any]) -> bool:
        """Check if a phase is completed. Optimize phase always runs."""
        # Optimize phase must always run (never skip)
        if phase == "optimize":
            return False
        return phase in state.get("completed_phases", [])

    def _mark_phase_completed(self, phase: str, state: Dict[str, Any]) -> None:
        """Mark a phase as completed."""
        if phase not in state.get("completed_phases", []):
            state.setdefault("completed_phases", []).append(phase)

        if "phases" not in state:
            state["phases"] = {}

        from datetime import datetime

        state["phases"][phase] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
        }

    def run(self) -> bool:
        """Run the full deployment pipeline."""
        self.logger.info(f"Starting orchestration for {self.model_dir}")

        state = self._load_state()

        for phase in self.phases:
            if self._is_phase_completed(phase, state):
                self.logger.info(
                    f"[INFO] Phase '{phase}' already completed, skipping..."
                )
                continue

            self.logger.info(f"[PHASE] Running: {phase}")

            success = self._run_phase(phase)

            if success:
                self._mark_phase_completed(phase, state)
                self._save_state(state)
                self.logger.info(f"[OK] Completed phase: {phase}")
            else:
                self.logger.error(f"[ERROR] Phase {phase} failed")
                return False

        self.logger.info("All phases completed successfully!")
        return True

    def _run_phase(self, phase: str) -> bool:
        """Execute a single phase."""
        try:
            if phase == "download":
                return self._run_download_phase()
            elif phase == "create_venv":
                return self._run_venv_phase()
            elif phase == "generate_config":
                return self._run_config_phase()
            elif phase == "optimize":
                return self._run_optimize_phase()
            else:
                self.logger.warning(f"Unknown phase: {phase}")
                return False
        except Exception as e:
            self.logger.error(f"Phase {phase} exception: {e}")
            return False

    def _run_download_phase(self) -> bool:
        """Run model download phase."""
        agent = PhaseAgent(
            model_dir=str(self.model_dir),
            max_retries=self.config.system.max_retries,
        )
        result = agent.run({"phase": "download"})
        return result.success

    def _run_venv_phase(self) -> bool:
        """Run virtual environment creation phase."""
        agent = PhaseAgent(
            model_dir=str(self.model_dir),
            max_retries=self.config.system.max_retries,
        )
        result = agent.run({"phase": "create_venv"})
        return result.success

    def _run_config_phase(self) -> bool:
        """Run configuration generation phase."""
        agent = PhaseAgent(
            model_dir=str(self.model_dir),
            max_retries=self.config.system.max_retries,
        )
        result = agent.run({"phase": "generate_config"})
        return result.success

    def _run_optimize_phase(self) -> bool:
        """Run optimization phase using intelligent V2 agent."""
        # Use new intelligent optimizer that calculates capacity from first principles
        agent = ExperimentAgentV2(
            model_dir=str(self.model_dir),
            max_retries=self.config.experiment.max_runs,
            verbose=True,
        )

        result = agent.run({})
        return result.success


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Orchestrate model deployment")
    parser.add_argument("model_dir", help="Path to model directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    orchestrator = Orchestrator(args.model_dir, args.verbose)
    success = orchestrator.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
