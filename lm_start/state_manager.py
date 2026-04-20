#!/usr/bin/env python3
"""
State Manager for setup_model.sh phases.

Manages setup state persistence, resumption, and phase tracking.
State is saved to setup_state.json in the model directory.

Usage:
    from state_manager import StateManager

    sm = StateManager(model_dir)
    sm.start_phase("download")
    # ... do work ...
    sm.complete_phase("download")

    # Check status
    if sm.can_resume():
        sm.resume()
"""

import json
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any


class PhaseStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Phase(str, Enum):
    INIT = "init"
    FETCH_INFO = "fetch_info"
    DOWNLOAD = "download"
    VENV = "venv"
    SMOKE_TEST = "smoke_test"
    EXTRACT_VLLM_CONFIG = "extract_vllm_config"
    GENERATE_CONFIG = "generate_config"
    OPTIMIZE = "optimize"
    FINALIZE = "finalize"


PHASE_ORDER = [
    Phase.INIT,
    Phase.FETCH_INFO,
    Phase.DOWNLOAD,
    Phase.VENV,
    Phase.SMOKE_TEST,
    Phase.EXTRACT_VLLM_CONFIG,
    Phase.GENERATE_CONFIG,
    Phase.OPTIMIZE,
    Phase.FINALIZE,
]


class StateManager:
    STATE_FILE = "setup_state.json"

    def __init__(self, model_dir: str, model_id: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.state_file = self.model_dir / self.STATE_FILE
        self.model_id = model_id
        self._state: Dict[str, Any] = {}

        if self.state_file.exists():
            self._load()
        elif model_id:
            self._init_state()

    def _init_state(self) -> None:
        now = datetime.now().isoformat()
        self._state = {
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "created_at": now,
            "updated_at": now,
            "current_phase": None,
            "phases": {
                phase.value: {
                    "status": PhaseStatus.PENDING.value,
                    "started_at": None,
                    "completed_at": None,
                    "error": None,
                    "metadata": {},
                }
                for phase in Phase
            },
            "tmux_sessions": {},
            "can_resume": False,
            "completed": False,
        }

    def _load(self) -> None:
        with open(self.state_file) as f:
            self._state = json.load(f)
        self.model_id = self._state.get("model_id")

    def _save(self) -> None:
        self._state["updated_at"] = datetime.now().isoformat()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def get_state(self) -> Dict[str, Any]:
        return self._state.copy()

    def start_phase(self, phase: Phase, metadata: Optional[Dict] = None) -> None:
        phase_key = phase.value if isinstance(phase, Phase) else phase
        now = datetime.now().isoformat()

        self._state["phases"][phase_key]["status"] = PhaseStatus.IN_PROGRESS.value
        self._state["phases"][phase_key]["started_at"] = now
        self._state["current_phase"] = phase_key

        if metadata:
            self._state["phases"][phase_key]["metadata"].update(metadata)

        self._save()

    def complete_phase(self, phase: Phase, metadata: Optional[Dict] = None) -> None:
        phase_key = phase.value if isinstance(phase, Phase) else phase
        now = datetime.now().isoformat()

        self._state["phases"][phase_key]["status"] = PhaseStatus.COMPLETED.value
        self._state["phases"][phase_key]["completed_at"] = now

        if metadata:
            self._state["phases"][phase_key]["metadata"].update(metadata)

        current_idx = PHASE_ORDER.index(Phase(phase_key))
        if current_idx < len(PHASE_ORDER) - 1:
            self._state["current_phase"] = PHASE_ORDER[current_idx + 1].value
        else:
            self._state["completed"] = True
            self._state["current_phase"] = None

        self._save()

    def fail_phase(self, phase: Phase, error: str) -> None:
        phase_key = phase.value if isinstance(phase, Phase) else phase

        self._state["phases"][phase_key]["status"] = PhaseStatus.FAILED.value
        self._state["phases"][phase_key]["error"] = error

        self._save()

    def skip_phase(self, phase: Phase, reason: str = "") -> None:
        phase_key = phase.value if isinstance(phase, Phase) else phase

        self._state["phases"][phase_key]["status"] = PhaseStatus.SKIPPED.value
        self._state["phases"][phase_key]["metadata"]["skip_reason"] = reason

        current_idx = PHASE_ORDER.index(Phase(phase_key))
        if current_idx < len(PHASE_ORDER) - 1:
            self._state["current_phase"] = PHASE_ORDER[current_idx + 1].value

        self._save()

    def get_phase_status(self, phase: Phase) -> PhaseStatus:
        phase_key = phase.value if isinstance(phase, Phase) else phase
        return PhaseStatus(self._state["phases"][phase_key]["status"])

    def get_current_phase(self) -> Optional[str]:
        return self._state.get("current_phase")

    def get_next_pending_phase(self) -> Optional[Phase]:
        for phase in PHASE_ORDER:
            if (
                self._state["phases"][phase.value]["status"]
                == PhaseStatus.PENDING.value
            ):
                return phase
        return None

    def can_resume(self) -> bool:
        if self._state.get("completed"):
            return False

        in_progress = any(
            p["status"] == PhaseStatus.IN_PROGRESS.value
            for p in self._state["phases"].values()
        )

        has_pending = any(
            p["status"] in (PhaseStatus.PENDING.value, PhaseStatus.IN_PROGRESS.value)
            for p in self._state["phases"].values()
        )

        return has_pending

    def get_resume_point(self) -> Optional[Phase]:
        current = self._state.get("current_phase")
        if current:
            current_phase = Phase(current)
            current_status = self.get_phase_status(current_phase)

            if current_status == PhaseStatus.IN_PROGRESS:
                return current_phase
            elif current_status == PhaseStatus.FAILED:
                return current_phase

        return self.get_next_pending_phase()

    def register_tmux_session(self, session_name: str, purpose: str) -> None:
        self._state["tmux_sessions"][session_name] = {
            "purpose": purpose,
            "created_at": datetime.now().isoformat(),
        }
        self._save()

    def unregister_tmux_session(self, session_name: str) -> None:
        self._state["tmux_sessions"].pop(session_name, None)
        self._save()

    def get_tmux_sessions(self) -> Dict[str, Any]:
        return self._state.get("tmux_sessions", {})

    def is_completed(self) -> bool:
        return self._state.get("completed", False)

    def print_status(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"Setup State: {self.model_id}")
        print(f"Model Dir: {self.model_dir}")
        print(f"{'=' * 60}")

        status_icons = {
            PhaseStatus.PENDING: "⏳",
            PhaseStatus.IN_PROGRESS: "🔄",
            PhaseStatus.COMPLETED: "✅",
            PhaseStatus.FAILED: "❌",
            PhaseStatus.SKIPPED: "⏭️",
        }

        for phase in PHASE_ORDER:
            phase_data = self._state["phases"][phase.value]
            status = PhaseStatus(phase_data["status"])
            icon = status_icons.get(status, "?")

            current = (
                " <- CURRENT" if self._state.get("current_phase") == phase.value else ""
            )
            print(f"  {icon} {phase.value:20} {status.value}{current}")

            if phase_data.get("error"):
                print(f"      Error: {phase_data['error']}")
            if phase_data.get("metadata"):
                for k, v in phase_data["metadata"].items():
                    print(f"      {k}: {v}")

        if self._state.get("tmux_sessions"):
            print(f"\nTmux Sessions:")
            for name, data in self._state["tmux_sessions"].items():
                print(f"  - {name}: {data.get('purpose', 'unknown')}")

        print(f"\nCan Resume: {self.can_resume()}")
        print(f"Completed: {self.is_completed()}")
        print(f"{'=' * 60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Manage setup state")
    parser.add_argument("model_dir", help="Model directory")
    parser.add_argument("--status", action="store_true", help="Print current status")
    parser.add_argument("--init", metavar="MODEL_ID", help="Initialize state for model")
    parser.add_argument("--start", metavar="PHASE", help="Start a phase")
    parser.add_argument("--complete", metavar="PHASE", help="Complete a phase")
    parser.add_argument("--fail", metavar="PHASE:ERROR", help="Mark phase as failed")
    parser.add_argument("--skip", metavar="PHASE:REASON", help="Skip a phase")
    parser.add_argument("--can-resume", action="store_true", help="Check if can resume")
    parser.add_argument("--resume-point", action="store_true", help="Get resume point")
    parser.add_argument(
        "--phase-status", metavar="PHASE", help="Get status of a specific phase"
    )
    parser.add_argument(
        "--register-tmux",
        nargs=2,
        metavar=("NAME", "PURPOSE"),
        help="Register tmux session",
    )

    args = parser.parse_args()

    sm = StateManager(args.model_dir, args.init)

    if args.init:
        sm._save()
        print(f"Initialized state for {args.init}")

    if args.status:
        sm.print_status()

    if args.start:
        sm.start_phase(Phase(args.start))
        print(f"Started phase: {args.start}")

    if args.complete:
        sm.complete_phase(Phase(args.complete))
        print(f"Completed phase: {args.complete}")

    if args.fail:
        phase, error = args.fail.split(":", 1)
        sm.fail_phase(Phase(phase), error)
        print(f"Failed phase: {phase} - {error}")

    if args.skip:
        parts = args.skip.split(":", 1)
        phase = parts[0]
        reason = parts[1] if len(parts) > 1 else ""
        sm.skip_phase(Phase(phase), reason)
        print(f"Skipped phase: {phase}")

    if args.can_resume:
        print(sm.can_resume())

    if args.resume_point:
        point = sm.get_resume_point()
        print(point.value if point else None)

    if args.phase_status:
        status = sm.get_phase_status(Phase(args.phase_status))
        print(status.value)

    if args.register_tmux:
        name, purpose = args.register_tmux
        sm.register_tmux_session(name, purpose)
        print(f"Registered tmux session: {name}")


if __name__ == "__main__":
    main()
