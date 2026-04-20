"""
State management for model setup phases.

Provides a clean interface for saving/loading setup state and resuming interrupted setups.
Uses the existing StateManager from state_manager.py.
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

from lm_start.state_manager import StateManager, Phase, PhaseStatus, PHASE_ORDER


class PhaseRunner:
    """
    Phase runner that provides a clean interface for executing setup phases.

    Handles state persistence, resumption, and phase execution with proper
    error handling and logging.
    """

    STATE_FILE = "setup_state.json"

    def __init__(self, model_dir: str, model_id: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.model_id = model_id
        self._sm: Optional[StateManager] = None

    @property
    def state_manager(self) -> StateManager:
        """Lazy-load state manager."""
        if self._sm is None:
            # Check if we need to initialize
            state_file = self.model_dir / self.STATE_FILE
            if state_file.exists():
                self._sm = StateManager(str(self.model_dir))
            elif self.model_id:
                self._sm = StateManager(str(self.model_dir), self.model_id)
                self._sm._save()
            else:
                raise ValueError(
                    "Either state file must exist or model_id must be provided"
                )
        return self._sm

    def init(self, model_id: str) -> "PhaseRunner":
        """Initialize state for a new model setup."""
        self.model_id = model_id
        self._sm = StateManager(str(self.model_dir), model_id)
        self._sm._save()
        return self

    def load(self) -> "PhaseRunner":
        """Load existing state from model directory."""
        state_file = self.model_dir / self.STATE_FILE
        if not state_file.exists():
            raise FileNotFoundError(f"No state file found at {state_file}")
        self._sm = StateManager(str(self.model_dir))
        return self

    @classmethod
    def from_model_dir(cls, model_dir: str) -> "PhaseRunner":
        """Create a PhaseRunner from model directory (loads existing or creates new)."""
        runner = cls(model_dir)
        state_file = runner.model_dir / cls.STATE_FILE
        if state_file.exists():
            runner._sm = StateManager(model_dir)
        return runner

    # Phase management methods
    def start_phase(
        self, phase: Phase, metadata: Optional[Dict] = None
    ) -> "PhaseRunner":
        """Mark a phase as started."""
        self.state_manager.start_phase(phase, metadata)
        return self

    def complete_phase(
        self, phase: Phase, metadata: Optional[Dict] = None
    ) -> "PhaseRunner":
        """Mark a phase as completed."""
        self.state_manager.complete_phase(phase, metadata)
        return self

    def fail_phase(self, phase: Phase, error: str) -> "PhaseRunner":
        """Mark a phase as failed."""
        self.state_manager.fail_phase(phase, error)
        return self

    def skip_phase(self, phase: Phase, reason: str = "") -> "PhaseRunner":
        """Skip a phase with a reason."""
        self.state_manager.skip_phase(phase, reason)
        return self

    def get_phase_status(self, phase: Phase) -> PhaseStatus:
        """Get status of a specific phase."""
        return self.state_manager.get_phase_status(phase)

    def is_phase_completed(self, phase: Phase) -> bool:
        """Check if a phase is already completed."""
        return self.get_phase_status(phase) == PhaseStatus.COMPLETED

    def is_phase_skipped(self, phase: Phase) -> bool:
        """Check if a phase was skipped."""
        return self.get_phase_status(phase) == PhaseStatus.SKIPPED

    # Resumption methods
    def can_resume(self) -> bool:
        """Check if setup can be resumed."""
        return self.state_manager.can_resume()

    def get_resume_point(self) -> Optional[Phase]:
        """Get the phase to resume from."""
        return self.state_manager.get_resume_point()

    def get_current_phase(self) -> Optional[str]:
        """Get the current phase."""
        return self.state_manager.get_current_phase()

    def is_completed(self) -> bool:
        """Check if setup is fully completed."""
        return self.state_manager.is_completed()

    # Status methods
    def print_status(self) -> None:
        """Print detailed status."""
        self.state_manager.print_status()

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of setup status."""
        state = self.state_manager.get_state()
        return {
            "model_id": state.get("model_id"),
            "model_dir": state.get("model_dir"),
            "current_phase": state.get("current_phase"),
            "can_resume": self.can_resume(),
            "completed": self.is_completed(),
            "phases": {
                phase.value: state.get("phases", {}).get(phase.value, {}).get("status")
                for phase in PHASE_ORDER
            },
        }


def create_state(model_dir: str, model_id: str) -> PhaseRunner:
    """Create a new state for model setup."""
    runner = PhaseRunner(model_dir)
    return runner.init(model_id)


def load_state(model_dir: str) -> PhaseRunner:
    """Load existing state from model directory."""
    return PhaseRunner(model_dir).load()


def can_resume(model_dir: str) -> bool:
    """Check if a setup can be resumed."""
    try:
        state_file = Path(model_dir) / PhaseRunner.STATE_FILE
        if not state_file.exists():
            return False
        sm = StateManager(model_dir)
        return sm.can_resume()
    except Exception:
        return False


def get_resume_point(model_dir: str) -> Optional[Phase]:
    """Get the phase to resume from."""
    try:
        sm = StateManager(model_dir)
        return sm.get_resume_point()
    except Exception:
        return None


def get_next_pending_phase(model_dir: str) -> Optional[Phase]:
    """Get the next pending phase."""
    try:
        sm = StateManager(model_dir)
        return sm.get_next_pending_phase()
    except Exception:
        return None
