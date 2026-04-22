"""
OpenCode Phase Recovery Agent
Simple agent for phase-level error recovery using OpenCode CLI.
"""

import json
import subprocess
import sys
import os
import time
import uuid
from pathlib import Path

from .opencode import OPENCODE_BINARY

# Phase-specific contexts for building prompts
PHASE_CONTEXTS = {
    "venv": """Python virtual environment creation failed.
Common issues to check:
- Python version compatibility (python3 --version)
- pip installation status and permissions
- Disk space availability (df -h)
- Package installation errors
- Virtual environment conflicts (existing venv directory)
- Network issues for package downloads""",
    "download": """Model or data download failed.
Common issues to check:
- Network connectivity (curl/wget test)
- HuggingFace authentication (hf_token valid)
- Disk space availability
- URL validity and accessibility
- Partial download cleanup
- Rate limiting issues""",
    "init": """Project initialization failed.
Common issues to check:
- Directory permissions (write access)
- Disk space
- Missing base files or templates
- Configuration file issues
- Ownership problems""",
    "generate_config": """Configuration generation failed.
Common issues to check:
- Missing required input files
- Invalid configuration values
- Template file issues
- Parsing errors in config files
- Missing dependencies""",
    "smoke_test": """vLLM serve smoke test failed.

CRITICAL: The model has its OWN virtual environment at {model_dir}/.venv
You CAN and SHOULD upgrade packages in this venv to fix issues!

FILES YOU CAN ACCESS AND MODIFY:
- Config: {model_dir}/smoke_test_config.yaml - MODIFY THIS FILE to fix issues
- Logs: {model_dir}/vllm_smoke_test.log - READ THIS to understand the error
- Extracted Config: {model_dir}/vllm_extracted_config.json - READ THIS for model info
- Venv Python: {model_dir}/.venv/bin/python
- Venv Pip: {model_dir}/.venv/bin/pip

IMPORTANT - CAN FIX BY UPGRADING PACKAGES:
If error mentions "Transformers does not recognize this architecture" or model type issues:
- Run: {model_dir}/.venv/bin/pip install --upgrade transformers
- Or: {model_dir}/.venv/bin/pip install git+https://github.com/huggingface/transformers.git

If error mentions CUDA or torch issues:
- Run: {model_dir}/.venv/bin/pip install --upgrade torch torchvision

The smoke test reads vLLM arguments from smoke_test_config.yaml.
Modify the 'vllm_args' section OR upgrade packages to fix startup issues.

Example smoke_test_config.yaml structure:
smoke_test:
  vllm_args:
    max_model_len: 1024
    tensor_parallel_size: 1
    dtype: "auto"
    trust_remote_code: true
    enable_chunked_prefill: true

Common fixes:
1. UPGRADE TRANSFORMERS: {model_dir}/.venv/bin/pip install --upgrade transformers
2. trust_remote_code: false -> true (for custom models like Kimi, DeepSeek)
3. max_model_len: 1024 -> 512 (if OOM during load)
4. Add dtype: "bfloat16" (for precision issues)
5. tensor_parallel_size: 1 -> 8 (if model needs more GPUs)

Read the error log first, then EITHER upgrade packages OR modify config to fix the specific error.""",
    "optimize": """vLLM optimization failed.

CRITICAL: The model has its OWN virtual environment at {model_dir}/.venv
You CAN and SHOULD modify files in this directory to fix issues!

FILES YOU CAN ACCESS AND MODIFY:
- Optimized Config: {model_dir}/optimized_config.json - MODIFY THIS FILE
- Best Config: {model_dir}/vllm_config_best.json - MODIFY THIS FILE
- Runs Dir: {model_dir}/.runs/ - Contains experiment results
- Smoke Config: {model_dir}/smoke_test_config.yaml - May need adjustment

IMPORTANT - OPTIMIZATION FIXES:
If optimization experiments keep failing:
1. Check vllm_config_best.json for working configuration
2. Lower max_model_len in smoke_test_config.yaml
3. Reduce tensor_parallel_size if OOM errors
4. Add enforce_eager: true to disable CUDA graphs
5. Change dtype from bfloat16 to float16 or auto

If no experiments can run:
- The base vLLM config is broken
- Fix smoke_test_config.yaml first
- Then retry optimization

Common optimization issues:
1. OOM errors -> Reduce max_model_len, increase tensor_parallel_size
2. Slow performance -> Adjust attention backend, enable chunked prefill
3. Context length too low -> Increase gradually, test stability
4. CUDA errors -> Add enforce_eager: true, disable CUDA graphs

Read the error logs in .runs/ directory to understand what failed.
Then modify smoke_test_config.yaml or optimized_config.json to fix issues.""",
}

OPENCODE_BIN = str(OPENCODE_BINARY)


class OpenCodePhaseAgent:
    """Agent that uses OpenCode CLI for phase recovery."""

    def __init__(self, model_dir: str, phase: str, error: str, max_retries: int = 3):
        self.model_dir = Path(model_dir)
        self.phase = phase
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
                capture_output=True,
                text=True,
                timeout=10,
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
                capture_output=True,
                text=True,
                timeout=10,
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
        import json
        from datetime import datetime

        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
        sessions = []
        if self.sessions_file.exists():
            with open(self.sessions_file) as f:
                sessions = json.load(f)
        sessions.append(
            {
                "session_id": session_id,
                "agent_type": "recovery",
                "phase": self.phase,
                "model_dir": str(self.model_dir),
                "timestamp": datetime.now().isoformat(),
                "opencode_cmd": f"{OPENCODE_BIN} --session {session_id}",
                "details": details or {},
            }
        )
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

    def _consult_opencode(self) -> dict:
        """Consult OpenCode CLI for a fix."""
        prompt = self._build_prompt()

        # Save prompt to temp file
        prompt_file = self.model_dir / ".opencode" / f"phase_recovery_{self.phase}.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(prompt)

        # Run opencode
        session_title = f"PhaseRecovery-{self.phase}-{uuid.uuid4().hex[:8]}"

        cmd = [
            OPENCODE_BIN,
            "run",
            "--model",
            "Grid/kimi-latest",
            "--agent",
            "build",
            "--dir",
            str(self.model_dir),
            "--title",
            session_title,
            "--file",
            str(prompt_file),
        ]

        time.sleep(2)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            time.sleep(2)
            session_id = self._find_session_by_title(session_title)
            if not session_id:
                session_id = self._extract_session_id(result.stdout + result.stderr)
            if not session_id:
                session_id = self._get_latest_opencode_session("recovery")
            if session_id:
                self._save_session(
                    session_id,
                    details={"attempt": self.attempts, "success": False},
                )

            return self._parse_response(result.stdout)

        except subprocess.TimeoutExpired:
            return {"success": False, "message": "OpenCode timeout"}
        except Exception as e:
            return {"success": False, "message": f"OpenCode error: {e}"}

    def _load_model_context(self) -> dict:
        import json
        import yaml

        context = {
            "model_id": "unknown",
            "gpu_count": "unknown",
            "gpu_memory": "unknown",
            "gpu_name": "unknown",
            "cuda_version": "unknown",
            "current_config": "No config file found",
        }

        model_info_file = (
            self.model_dir / ".llm-context" / "model-context" / "model_info.json"
        )
        if model_info_file.exists():
            try:
                with open(model_info_file) as f:
                    model_info = json.load(f)
                context["model_id"] = model_info.get("model_id", "unknown")
            except:
                pass

        device_config_file = (
            self.model_dir / ".llm-context" / "model-context" / "device_config.json"
        )
        if device_config_file.exists():
            try:
                with open(device_config_file) as f:
                    device_config = json.load(f)
                gpus = device_config.get("gpus", {})
                context["gpu_count"] = str(gpus.get("count", "unknown"))
                context["gpu_memory"] = str(gpus.get("memory_gb_per_gpu", "unknown"))
                gpu_names = gpus.get("names", ["unknown"])
                context["gpu_name"] = gpu_names[0] if gpu_names else "unknown"
                context["cuda_version"] = device_config.get("cuda", {}).get(
                    "version", "unknown"
                )
            except:
                pass

        smoke_config_file = self.model_dir / "smoke_test_config.yaml"
        if smoke_config_file.exists():
            try:
                with open(smoke_config_file) as f:
                    smoke_config = yaml.safe_load(f)
                vllm_args = smoke_config.get("smoke_test", {}).get("vllm_args", {})
                if vllm_args:
                    config_lines = []
                    for k, v in vllm_args.items():
                        config_lines.append(f"  {k}: {v}")
                    context["current_config"] = "\n".join(config_lines)
            except:
                pass

        return context

    def _build_prompt(self) -> str:
        model_ctx = self._load_model_context()

        context = PHASE_CONTEXTS.get(
            self.phase, "Phase failed. Analyze error and suggest fix."
        ).format(
            model_dir=str(self.model_dir),
            model_id=model_ctx["model_id"],
            gpu_count=model_ctx["gpu_count"],
            gpu_memory=model_ctx["gpu_memory"],
            gpu_name=model_ctx["gpu_name"],
            cuda_version=model_ctx["cuda_version"],
            current_config=model_ctx["current_config"],
        )

        return f"""=== PHASE RECOVERY AGENT ===

PHASE: {self.phase}
MODEL DIR: {self.model_dir}
MODEL ID: {model_ctx["model_id"]}
HARDWARE: {model_ctx["gpu_count"]}x {model_ctx["gpu_name"]} ({model_ctx["gpu_memory"]}GB each)
CUDA: {model_ctx["cuda_version"]}
ATTEMPT: {self.attempts}/{self.max_retries}

CURRENT CONFIG VALUES (what to change FROM):
{model_ctx["current_config"]}

ERROR OUTPUT:
{self.error}

CONTEXT:
{context}

YOUR TASK:
1. Analyze the error above
2. Determine the root cause
3. Suggest ONE specific fix command or file modification
4. Return ONLY JSON with this format:

{{
  "analysis": "Brief error analysis",
  "root_cause": "What caused it",
  "fix": {{
    "command": "The exact shell command to run",
    "description": "What this command does",
    "requires_sudo": false
  }},
  "fallback": "Alternative if first fix fails"
}}

Be specific. The command should be runnable directly.
"""

    def _parse_response(self, output: str) -> dict:
        """Parse OpenCode response."""
        import re

        # Try to find JSON in output
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if not json_match:
            return {"success": False, "message": "No JSON found in response"}

        try:
            data = json.loads(json_match.group())
            fix = data.get("fix", {})
            command = fix.get("command", "")

            if not command:
                return {"success": False, "message": "No fix command suggested"}

            # Execute the command
            print(f"[OpenCodePhaseAgent] Executing: {command}")

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.model_dir),
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Fix applied: {fix.get('description', 'Success')}",
                    "command": command,
                }
            else:
                return {
                    "success": False,
                    "message": f"Command failed: {result.stderr[:200]}",
                }

        except json.JSONDecodeError as e:
            return {"success": False, "message": f"JSON parse error: {e}"}
        except Exception as e:
            return {"success": False, "message": f"Error: {e}"}


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
