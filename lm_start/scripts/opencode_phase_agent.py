"""
OpenCode Phase Recovery Agent
Agentic recovery using OpenCode CLI - reads logs and modifies config directly.
"""

import json
import subprocess
import sys
import os
import time
import uuid
from pathlib import Path

from lm_start.scripts.opencode import OPENCODE_BINARY

OPENCODE_BIN = str(OPENCODE_BINARY)

# Prompts YAML path
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
PHASE_RECOVERY_YAML = PROMPTS_DIR / "phase_recovery_agent.yaml"


def _load_phase_prompt(phase: str) -> str:
    """Load phase-specific prompt from YAML file."""
    try:
        import yaml
        with open(PHASE_RECOVERY_YAML) as f:
            prompts = yaml.safe_load(f)
        key = f"{phase}_system"
        return prompts.get(key, prompts.get("smoke_test_system", ""))
    except Exception:
        return ""


class OpenCodePhaseAgent:
    """Agent that uses OpenCode CLI for phase recovery."""

    def __init__(self, model_dir: str, phase: str, error: str, max_retries: int = 3):
        self.model_dir = Path(model_dir)
        self.phase = phase
        self.opencode_dir = self.model_dir / ".opencode"
        self.error = error
        self.max_retries = max_retries
        self.attempts = 0
        self.sessions_file = self.model_dir / ".sessions" / "sessions.json"

    def _extract_session_id(self, output: str):
        import re
        match = re.search(r"ses_[a-zA-Z0-9]+", output)
        return match.group(0) if match else None

    def _get_latest_opencode_session(self, title_hint: str = None):
        try:
            result = subprocess.run(
                [OPENCODE_BIN, "session", "list"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return None
            lines = result.stdout.strip().split("\n")
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 3:
                    session_id = parts[0]
                    title = " ".join(parts[1:-1])
                    if session_id.startswith("ses_"):
                        if title_hint and title_hint.lower() in title.lower():
                            return session_id
                        elif not title_hint:
                            return session_id
            return None
        except Exception:
            return None

    def _find_session_by_title(self, title_prefix: str):
        try:
            result = subprocess.run(
                [OPENCODE_BIN, "session", "list"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return None
            lines = result.stdout.strip().split("\n")
            for line in reversed(lines[1:]):
                if title_prefix in line:
                    parts = line.split()
                    if parts and parts[0].startswith("ses_"):
                        return parts[0]
            return None
        except Exception:
            return None

    def _save_session(self, session_id: str, details: dict = None):
        if not session_id:
            return
        from datetime import datetime
        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
        sessions = []
        if self.sessions_file.exists():
            with open(self.sessions_file) as f:
                sessions = json.load(f)
        sessions.append({
            "session_id": session_id,
            "agent_type": "recovery",
            "phase": self.phase,
            "model_dir": str(self.model_dir),
            "timestamp": datetime.now().isoformat(),
            "opencode_cmd": f"{OPENCODE_BIN} --session {session_id}",
            "details": details or {},
        })
        with open(self.sessions_file, "w") as f:
            json.dump(sessions, f, indent=2)
        print(f"[SESSION] Saved: {session_id} (recovery/{self.phase})")

    def run(self) -> dict:
        """Run recovery attempts."""
        print(f"[OpenCodePhaseAgent] Handling failure in phase: {self.phase}")
        print(f"[OpenCodePhaseAgent] Error: {self.error[:200]}...")

        while self.attempts < self.max_retries:
            self.attempts += 1
            print(f"[OpenCodePhaseAgent] Attempt {self.attempts}/{self.max_retries}")

            result = self._consult_opencode()

            if result.get("success"):
                return {"success": True, "message": result.get("message", "Fixed")}

            if self.attempts >= self.max_retries:
                break

        return {"success": False, "message": "OpenCode agent could not fix the issue"}

    def _load_model_context(self) -> dict:
        import yaml as _yaml

        context = {
            "model_id": "unknown",
            "gpu_count": "unknown",
            "gpu_memory": "unknown",
            "gpu_name": "unknown",
            "cuda_version": "unknown",
            "current_config_yaml": "No config file found",
            "model_params_b": "unknown",
            "model_size_gb": "unknown",
            "is_moe": "unknown",
            "architecture": "unknown",
            "max_position_embeddings": "unknown",
            "total_gpu_memory_gb": "unknown",
        }

        model_info_file = self.model_dir / ".llm-context" / "model-context" / "model_info.json"
        if model_info_file.exists():
            try:
                with open(model_info_file) as f:
                    model_info = json.load(f)
                context["model_id"] = model_info.get("model_id", "unknown")
            except Exception:
                pass

        device_config_file = self.model_dir / ".llm-context" / "model-context" / "device_config.json"
        if device_config_file.exists():
            try:
                with open(device_config_file) as f:
                    device_config = json.load(f)
                gpus = device_config.get("gpus", {})
                gpu_count = gpus.get("count", 0)
                gpu_mem = gpus.get("memory_gb_per_gpu", 0)
                context["gpu_count"] = str(gpu_count)
                context["gpu_memory"] = str(gpu_mem)
                gpu_names = gpus.get("names", ["unknown"])
                context["gpu_name"] = gpu_names[0] if gpu_names else "unknown"
                context["cuda_version"] = device_config.get("cuda", {}).get("version", "unknown")
                total = gpu_count * gpu_mem if isinstance(gpu_count, int) and isinstance(gpu_mem, (int, float)) else "unknown"
                context["total_gpu_memory_gb"] = str(total)
            except Exception:
                pass

        extracted_config_file = self.model_dir / "vllm_extracted_config.json"
        if not extracted_config_file.exists():
            extracted_config_file = self.model_dir / ".llm-context" / "model-context" / "vllm_extracted_config.json"
        if extracted_config_file.exists():
            try:
                with open(extracted_config_file) as f:
                    extracted = json.load(f)
                mem = extracted.get("memory_estimate", {})
                context["model_params_b"] = f"{mem.get('parameters_b', 'unknown'):.1f}" if isinstance(mem.get('parameters_b'), float) else str(mem.get('parameters_b', 'unknown'))
                actual_size = mem.get('actual_size_gb') or mem.get('total_size_gb')
                context["model_size_gb"] = f"{actual_size:.1f}" if isinstance(actual_size, (int, float)) else "unknown"
                context["is_moe"] = str(extracted.get("is_moe", False))
                hf_cfg = extracted.get("hf_config", {})
                archs = hf_cfg.get("architectures", ["unknown"])
                context["architecture"] = archs[0] if archs else "unknown"
                max_pos = hf_cfg.get("max_position_embeddings", 0) or hf_cfg.get("max_seq_len", 0) or hf_cfg.get("seq_length", 0)
                context["max_position_embeddings"] = str(max_pos) if max_pos else "unknown"
            except Exception:
                pass

        smoke_config_file = self.model_dir / "smoke_test_config.yaml"
        if smoke_config_file.exists():
            try:
                context["current_config_yaml"] = smoke_config_file.read_text()
            except Exception:
                pass

        # Last 2000 chars of log
        smoke_log = self.model_dir / "vllm_smoke_test.log"
        if smoke_log.exists():
            try:
                log_content = smoke_log.read_text()
                context["error_summary"] = log_content[-2000:] if len(log_content) > 2000 else log_content
            except Exception:
                context["error_summary"] = self.error
        else:
            context["error_summary"] = self.error

        return context

    def _build_prompt(self) -> str:
        model_ctx = self._load_model_context()

        # Load from YAML prompt file
        template = _load_phase_prompt(self.phase)

        if template:
            prompt = template.format(
                model_dir=str(self.model_dir),
                model_id=model_ctx["model_id"],
                gpu_count=model_ctx["gpu_count"],
                gpu_memory=model_ctx["gpu_memory"],
                gpu_name=model_ctx["gpu_name"],
                cuda_version=model_ctx["cuda_version"],
                total_gpu_memory_gb=model_ctx.get("total_gpu_memory_gb", "unknown"),
                model_params_b=model_ctx.get("model_params_b", "unknown"),
                model_size_gb=model_ctx.get("model_size_gb", "unknown"),
                is_moe=model_ctx.get("is_moe", "unknown"),
                architecture=model_ctx.get("architecture", "unknown"),
                max_position_embeddings=model_ctx.get("max_position_embeddings", "unknown"),
                attempt=self.attempts,
                max_retries=self.max_retries,
                current_config_yaml=model_ctx.get("current_config_yaml", ""),
                error_summary=model_ctx.get("error_summary", self.error),
            )
        else:
            # Minimal fallback
            prompt = f"""=== PHASE RECOVERY AGENT ===
PHASE: {self.phase}
MODEL DIR: {self.model_dir}
ERROR: {self.error[:500]}

Read the log files and fix the issue by modifying smoke_test_config.yaml directly.
"""

        return prompt

    def _consult_opencode(self) -> dict:
        """Consult OpenCode CLI for a fix — agent reads logs and writes fixes directly."""
        prompt = self._build_prompt()

        prompt_file = self.opencode_dir / f"phase_recovery_{self.phase}.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(prompt)

        session_title = f"PhaseRecovery-{self.phase}-{uuid.uuid4().hex[:8]}"

        cmd = [
            OPENCODE_BIN,
            "run",
            "--model", "Grid/kimi-latest",
            "--agent", "build",
            "--dir", str(self.model_dir),
            "--title", session_title,
            "recovery",
            "--file", str(prompt_file),
        ]

        time.sleep(2)

        # Record mtime of config before agent runs
        smoke_config = self.model_dir / "smoke_test_config.yaml"
        mtime_before = smoke_config.stat().st_mtime if smoke_config.exists() else 0

        try:
            from lm_start.core.config import get_config
            from lm_start.commands.env import get_credentials_env

            env = get_config().get_env_dict(str(self.model_dir / ".venv"))
            env.update(get_credentials_env())

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min — agent needs time to read files and write fixes
                env=env,
            )

            time.sleep(2)
            session_id = self._find_session_by_title(session_title)
            if not session_id:
                session_id = self._extract_session_id(result.stdout + result.stderr)
            if not session_id:
                session_id = self._get_latest_opencode_session("recovery")
            if session_id:
                self._save_session(session_id, details={"attempt": self.attempts})

            # Check if agent actually modified the config file
            mtime_after = smoke_config.stat().st_mtime if smoke_config.exists() else 0
            config_modified = mtime_after > mtime_before

            if config_modified:
                print(f"[OpenCodePhaseAgent] ✓ smoke_test_config.yaml was updated by agent")
                return {"success": True, "message": "Agent updated smoke_test_config.yaml"}

            # Fallback: check output for success indicators
            if result.returncode == 0 and result.stdout:
                output_lower = (result.stdout + result.stderr).lower()
                if any(kw in output_lower for kw in ["fixed", "updated", "modified", "written", "success"]):
                    return {"success": True, "message": "Agent reported fix applied"}

            print(f"[OpenCodePhaseAgent] Agent did not modify config (rc={result.returncode})")
            if result.stderr:
                print(f"[OpenCodePhaseAgent] stderr: {result.stderr[:200]}")
            return {"success": False, "message": "Agent did not modify config file"}

        except subprocess.TimeoutExpired:
            return {"success": False, "message": "OpenCode timeout (300s)"}
        except Exception as e:
            return {"success": False, "message": f"OpenCode error: {e}"}


def main():
    """CLI entry point."""
    if len(sys.argv) < 4:
        print("Usage: opencode_phase_agent.py <model_dir> <phase> <error>")
        sys.exit(1)

    model_dir = sys.argv[1]
    phase = sys.argv[2]
    error = sys.argv[3]

    agent = OpenCodePhaseAgent(model_dir, phase, error)
    result = agent.run()

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()


# Phase-specific contexts for building prompts
