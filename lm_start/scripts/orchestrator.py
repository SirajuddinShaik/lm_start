#!/usr/bin/env python3
"""
OpenCode Agentic Orchestrator

Implements the PLAN_AGENTIC.md architecture:
1. Master Planner Agent (opencode) - decides experiments
2. Experiment Executor (code) - runs vLLM
3. Run Summarizer Agent (opencode) - compresses logs

Sequential execution: Only one vLLM instance at a time (GPU constraint)
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
import yaml
from rich.console import Console as _Console

_console = _Console()

from lm_start.utils.model_utils import get_model_cache_path, read_model_config
from lm_start.utils.theoretical_calculator import HardwareConfig, ModelConfig
from lm_start.utils.profile_config_generator import ProfileConfigGenerator, ProfileType
from lm_start.utils.benchmark_runner import BenchmarkRunner
from lm_start.utils.prompt_manager import PromptManager
from lm_start.utils.log_parser import parse_vllm_logs
from lm_start.vllm_flag_validator import validate_vllm_flags
from lm_start.core.config import get_config
from lm_start.scripts.opencode import OPENCODE_BINARY


class OpenCodeAgenticOrchestrator:
    """
    Orchestrates the agentic optimization loop using OpenCode CLI.

    Flow:
    1. Spawn Planner Agent (opencode) → Get experiment queue
    2. For each experiment:
       a. Create run directory
       b. Spawn vLLM
       c. Run benchmark
       d. Save results
       e. Spawn Summarizer Agent (opencode) → Update master plan
    3. Repeat until optimization complete
    """

    def __init__(
        self,
        model_dir: str,
        model_id: str,
        goal: str = "maximize context",
        force: bool = False,
        min_experiments: int = 1,
        optimization_mode: str = "normal",
    ):
        self.model_dir = Path(model_dir)
        self.model_id = model_id
        self.goal = goal
        self.force = force
        self.min_experiments = min_experiments
        self.optimization_mode = optimization_mode
        self.experiments_run = 0
        self.opencode_bin = str(OPENCODE_BINARY)

        # Directories
        self.runs_dir = self.model_dir / ".runs"
        self.summaries_dir = self.model_dir / ".summaries"
        self.logs_dir = self.model_dir / ".logs"
        self.opencode_dir = self.model_dir / ".opencode"
        self.sessions_dir = self.model_dir / ".sessions"

        # State
        self.master_plan_file = self.summaries_dir / "run_master_plan.json"
        self.sessions_file = self.sessions_dir / "sessions.json"
        self.current_run_id = 0
        self.prompts = self._load_prompts()
        self.config = get_config()

        # Current plan context (insights from planner for current iteration)
        self._current_insights_read = ""
        self._current_knowledge_update = ""

    def setup_directories(self):
        """Create required directory structure."""
        for dir_path in [
            self.runs_dir,
            self.summaries_dir,
            self.logs_dir,
            self.opencode_dir,
            self.sessions_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _extract_session_id(self, output: str) -> Optional[str]:
        import re

        # Find ALL session IDs and return the LAST one (most recent)
        matches = re.findall(r"ses_[a-zA-Z0-9]+", output)
        return matches[-1] if matches else None

    def _opencode_db_path(self) -> Optional[Path]:
        db = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
        return db if db.exists() else None

    def _find_session_by_title(self, title_prefix: str) -> Optional[str]:
        db = self._opencode_db_path()
        if db:
            try:
                import sqlite3
                conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                row = conn.execute(
                    "SELECT id FROM session WHERE title = ? AND directory = ? ORDER BY time_created DESC LIMIT 1",
                    (title_prefix, str(self.model_dir)),
                ).fetchone()
                conn.close()
                if row:
                    return row[0]
            except Exception:
                pass
        # Fallback: parse opencode session list output
        try:
            result = subprocess.run(
                [self.opencode_bin, "session", "list"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if title_prefix in line:
                        parts = line.split()
                        if parts and parts[0].startswith("ses_"):
                            return parts[0]
        except Exception:
            pass
        return None

    def _get_latest_opencode_session(self, title_hint: str = None) -> Optional[str]:
        db = self._opencode_db_path()
        if db:
            try:
                import sqlite3
                conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                if title_hint:
                    row = conn.execute(
                        "SELECT id FROM session WHERE directory = ? AND title LIKE ? ORDER BY time_created DESC LIMIT 1",
                        (str(self.model_dir), f"%{title_hint}%"),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT id FROM session WHERE directory = ? ORDER BY time_created DESC LIMIT 1",
                        (str(self.model_dir),),
                    ).fetchone()
                conn.close()
                if row:
                    return row[0]
            except Exception:
                pass
        # Fallback: parse opencode session list output
        try:
            result = subprocess.run(
                [self.opencode_bin, "session", "list"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return None
            for line in reversed(result.stdout.strip().split("\n")):
                parts = line.split()
                if len(parts) >= 2 and parts[0].startswith("ses_"):
                    title = " ".join(parts[1:])
                    if title_hint and title_hint.lower() not in title.lower():
                        continue
                    return parts[0]
        except Exception:
            pass
        return None

    def _save_session(
        self,
        session_id: str,
        agent_type: str,
        run_id: str = None,
        phase: str = None,
        details: dict = None,
    ):
        if not session_id:
            return

        sessions = []
        if self.sessions_file.exists():
            with open(self.sessions_file) as f:
                sessions = json.load(f)

        # Remove existing session for same phase (override on rerun)
        if phase:
            sessions = [s for s in sessions if s.get("phase") != phase]

        session_entry = {
            "session_id": session_id,
            "agent_type": agent_type,
            "run_id": run_id,
            "phase": phase,
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "timestamp": datetime.now().isoformat(),
            "opencode_cmd": f"{self.opencode_bin} --session {session_id}",
            "details": details or {},
        }

        sessions.append(session_entry)

        with open(self.sessions_file, "w") as f:
            json.dump(sessions, f, indent=2)

        action = "Updated" if phase else "Saved"
        # Session saves are silent — visible via `lm-start session list`
        return session_id

    def _is_port_in_use(self, port: int) -> bool:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def _find_free_port(self, start_port: int = 8000, end_port: int = 9000) -> int:
        import socket

        for port in range(start_port, end_port + 1):
            if not self._is_port_in_use(port):
                return port
        raise RuntimeError(f"No free ports found in range {start_port}-{end_port}")

    def _get_process_using_port(self, port: int):
        import psutil

        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port and conn.pid:
                try:
                    return psutil.Process(conn.pid)
                except psutil.NoSuchProcess:
                    pass
        return None

    def _kill_existing_vllm(self, target_port: int = None):
        import psutil

        killed = []
        pids_to_kill = set()

        if target_port and self._is_port_in_use(target_port):
            _console.print(f"  [yellow]⚠[/yellow]  Port {target_port} is already in use")
            proc = self._get_process_using_port(target_port)
            if proc:
                _console.print(f"  [dim]Port {target_port} in use by PID {proc.pid} ({proc.name()})[/dim]")
                pids_to_kill.add(proc.pid)
            else:
                _console.print(f"  [yellow]⚠[/yellow]  Could not identify process using port {target_port}")
        else:
            return killed

        for pid in pids_to_kill:
            try:
                proc = psutil.Process(pid)
                _console.print(f"  [dim]ℹ[/dim]    Terminating process {pid} ({proc.name()})")
                proc.terminate()
                gone, alive = psutil.wait_procs([proc], timeout=10)
                if proc in alive:
                    _console.print(f"  [yellow]⚠[/yellow]  Process {pid} did not terminate, forcing kill")
                    proc.kill()
                    proc.wait(timeout=5)
                killed.append(pid)
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                _console.print(f"  [yellow]⚠[/yellow]  Error killing process {pid}: {e}")

        if killed:
            _console.print(f"  [green]✓[/green]  Killed {len(killed)} process(es): {killed}")
            time.sleep(3)

            # Verify port is freed
            if target_port:
                if self._is_port_in_use(target_port):
                    _console.print(f"  [red]✗[/red]  Port {target_port} is still in use!")
                else:
                    _console.print(f"  [green]✓[/green]  Port {target_port} is now free")

        return len(killed)

    def _check_gpu_memory(self, required_gb: float = 10.0) -> bool:
        """Check if sufficient GPU memory is available."""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return True

            free_memories = [
                float(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()
            ]
            min_free = min(free_memories) / 1024

            if min_free < required_gb:
                _console.print(
                    f"  [yellow]⚠[/yellow]  Low GPU memory: {min_free:.1f} GB free (need {required_gb} GB)"
                )
                return False

            _console.print(f"  [green]✓[/green]  GPU memory check passed: {min_free:.1f}GB free")
            return True
        except Exception as e:
            _console.print(f"  [yellow]⚠[/yellow]  Could not check GPU memory: {e}")
            return True

    def _setup_insights_file(self):
        insights_file = self.model_dir / "Insights.md"
        if not insights_file.exists():
            template_file = Path(__file__).parent / ".opencode" / "Insights.md"
            if template_file.exists():
                shutil.copy2(template_file, insights_file)
                print(f"Created Insights.md from template")
            else:
                insights_file.write_text(
                    f"# Insights for {self.model_id}\n\n## Hardware Capabilities\n\n## Available Backends\n\n## Working Configurations\n\n## Error Patterns\n"
                )
                print(f"Created minimal Insights.md")

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML files."""
        # Try scripts/prompts first, then fall back to lm_start/prompts
        scripts_prompts = Path(__file__).parent / "prompts"
        lm_start_prompts = Path(__file__).parent.parent / "prompts"
        prompts_dir = scripts_prompts if scripts_prompts.exists() else lm_start_prompts
        prompts = {}

        planner_file = prompts_dir / "opencode_planner.yaml"
        if planner_file.exists():
            with open(planner_file) as f:
                prompts["planner"] = yaml.safe_load(f)

        summarizer_file = prompts_dir / "opencode_summarizer.yaml"
        if summarizer_file.exists():
            with open(summarizer_file) as f:
                prompts["summarizer"] = yaml.safe_load(f)

        run_summarizer_file = prompts_dir / "run_summarizer.yaml"
        if run_summarizer_file.exists():
            with open(run_summarizer_file) as f:
                prompts["run_summarizer"] = yaml.safe_load(f)

        return prompts

    def load_model_context(self) -> Dict[str, Any]:
        """Load model info and device config."""
        context: Dict[str, Any] = {
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "goal": self.goal,
            "timestamp": datetime.now().isoformat(),
        }

        model_info_file = (
            self.model_dir / ".llm-context" / "model-context" / "model_info.json"
        )
        if model_info_file.exists():
            with open(model_info_file) as f:
                context["model_info"] = json.load(f)

        device_config_file = (
            self.model_dir / ".llm-context" / "model-context" / "device_config.json"
        )
        if device_config_file.exists():
            with open(device_config_file) as f:
                context["device_config"] = json.load(f)

        # Load HF model config
        cache_path = get_model_cache_path(self.model_id)
        if cache_path:
            model_config = read_model_config(str(cache_path))
            if model_config:
                context["model_config"] = model_config

        return context

    AGENT_FAILURE_EXIT_CODE = 2  # Exit code when planner fails to generate valid JSON

    def spawn_planner_agent(self) -> Dict[str, Any]:
        """
        Spawn Master Planner Agent via OpenCode CLI.

        Returns experiment queue from planner.
        Sets self._planner_failed flag if agent couldn't generate valid JSON.
        """
        self._planner_failed = False  # Reset flag

        context = self.load_model_context()
        experiment_history = self._load_experiment_history_from_runs()
        prompt = self._build_planner_prompt(context, experiment_history)

        # Save prompt to file for opencode
        prompt_file = self.opencode_dir / "planner_prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(prompt)

        session_title = (
            f"Planner-{self.model_id.replace('/', '-')}-{uuid.uuid4().hex[:8]}"
        )

        cmd = [
            self.opencode_bin,
            "run",
            "--model",
            "Grid/kimi-latest",
            "--agent",
            "build",
            "--dir",
            str(self.model_dir),
            "--title",
            session_title,
            "planner",
            "--file",
            str(prompt_file),
        ]

        _console.print(f"  [dim]Running planner agent...[/dim]")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=self.config.get_env_dict(str(self.model_dir / ".venv")),
            )

            debug_file = self.opencode_dir / "planner_output_debug.txt"
            with open(debug_file, "w") as f:
                f.write(f"STDOUT:\n{result.stdout}\n\n")
                f.write(f"STDERR:\n{result.stderr}\n\n")
                f.write(f"Return code: {result.returncode}\n")

            session_id = self._find_session_by_title(session_title)
            if not session_id:
                session_id = self._extract_session_id(result.stdout + result.stderr)
            if not session_id:
                session_id = self._get_latest_opencode_session("planner")
            if session_id:
                self._save_session(
                    session_id,
                    "planner",
                    phase="optimize",
                    details={
                        "experiments_count": len(
                            self._extract_json_from_output(result.stdout).get(
                                "experiments_queued", []
                            )
                        ),
                        "iteration": getattr(self, "_iteration", 1),
                    },
                )

            plan = self._extract_json_from_output(result.stdout)

            with open(self.master_plan_file, "w") as f:
                json.dump(plan, f, indent=2)

            # Check if planner failed to generate valid experiments
            if (
                plan.get("optimization_complete")
                and len(plan.get("experiments_queued", [])) == 0
                and plan.get("reasoning", "").startswith("Failed to parse")
            ):
                self._planner_failed = True
                _console.print(f"  [red]✗[/red]  Planner failed to generate valid JSON")
                return plan

            n = len(plan.get('experiments_queued', []))
            _console.print(f"  [green]✓[/green]  Planner: {n} experiment{'s' if n != 1 else ''} queued")
            if plan.get("reasoning"):
                _console.print(f"      {plan['reasoning'][:120]}", markup=False, highlight=False)

            # Save experiment reasoning/predictions
            for exp in plan.get("experiments_queued", []):
                if exp.get("reasoning") or exp.get("prediction"):
                    exp["_planner_context"] = {
                        "reasoning": exp.get("reasoning", ""),
                        "prediction": exp.get("prediction", {}),
                    }

            if plan.get("insights_read") or plan.get("knowledge_update"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                insights_file = self.opencode_dir / f"planner_insights_{timestamp}.json"
                insights_data = {
                    "timestamp": datetime.now().isoformat(),
                    "insights_read": plan.get("insights_read", ""),
                    "knowledge_update": plan.get("knowledge_update", ""),
                    "optimization_complete": plan.get("optimization_complete", False),
                }
                with open(insights_file, "w") as f:
                    json.dump(insights_data, f, indent=2)
                _console.print(f"  [dim]Planner insights saved[/dim]")

                if plan.get("insights_read"):
                    self._current_insights_read = plan["insights_read"]
                    self._append_insights_read_to_md(plan["insights_read"])
                else:
                    self._current_insights_read = ""

                if plan.get("knowledge_update"):
                    self._current_knowledge_update = plan["knowledge_update"]
                    self._append_planner_insights_to_md(plan["knowledge_update"])
                else:
                    self._current_knowledge_update = ""

            return plan

        except subprocess.TimeoutExpired:
            _console.print(f"  [yellow]⚠[/yellow]  Planner timed out")
            self._planner_failed = True
            sys.exit(self.AGENT_FAILURE_EXIT_CODE)
        except Exception as e:
            _console.print(f"  [red]✗[/red]  Planner error: {e}")
            self._planner_failed = True
            sys.exit(self.AGENT_FAILURE_EXIT_CODE)

    def _build_planner_prompt(self, context: Dict, history: List[Dict]) -> str:
        """Build dynamic planner prompt from context using external prompt files."""
        device = context.get("device_config", {})
        model = context.get("model_info", {})
        model_cfg = context.get("model_config", {})

        gpu_info = device.get("gpus", {})
        gpu_count = gpu_info.get("count", 0)
        gpu_names = gpu_info.get("names", ["unknown"])
        gpu_memory = gpu_info.get("memory_gb_per_gpu", 0)
        compute_capability = gpu_info.get("compute_capability", "unknown")

        model_id = model.get("model_id", "unknown")

        text_config = model_cfg.get("text_config", {})
        max_position = model_cfg.get("max_position_embeddings", 0) or text_config.get(
            "max_position_embeddings", 0
        )
        num_layers = model_cfg.get("num_hidden_layers", 0) or text_config.get(
            "num_hidden_layers", 0
        )
        hidden_size = model_cfg.get("hidden_size", 0) or text_config.get(
            "hidden_size", 0
        )
        num_attention_heads = model_cfg.get(
            "num_attention_heads", 0
        ) or text_config.get("num_attention_heads", 0)

        if not max_position:
            max_position = model.get("context_length", 0)
        if not num_layers:
            num_layers = model_cfg.get("num_layers", 0)
        if not hidden_size:
            hidden_size = model_cfg.get("d_model", 0)
        if not num_attention_heads:
            num_attention_heads = model_cfg.get("num_heads", 0)

        estimated_params = (
            (num_layers * hidden_size * hidden_size * 12) / 1e9
            if num_layers and hidden_size
            else 0
        )

        target_context = max_position if max_position else 32768
        calc_max_num_seqs = min(128, max(64, target_context // 256))
        calc_batched_tokens = min(32768, target_context / 4)
        calc_gpu_util = 0.90

        # Respect CUDA_VISIBLE_DEVICES - use visible count if set
        visible_devices = device.get("gpus", {}).get("visible_devices", "")
        if visible_devices and "," in str(visible_devices):
            # Count actual visible GPUs
            visible_count = len(str(visible_devices).split(","))
            effective_gpu_count = visible_count
        else:
            effective_gpu_count = gpu_count

        # Calculate model folder size
        model_folder_size_gb = 0
        try:
            import subprocess

            result = subprocess.run(
                ["du", "-sb", str(self.model_dir)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                size_bytes = int(result.stdout.split()[0])
                model_folder_size_gb = round(size_bytes / (1024**3), 1)
        except:
            pass

        # Calculate HF cache size (entire models--* directory including blobs)
        hf_cache_size_gb = 0
        try:
            cache_id = self.model_id.replace("/", "--")
            hf_home = os.environ.get("HF_HOME", "/data/.cache/huggingface")
            cache_root = Path(hf_home) / "hub" / f"models--{cache_id}"
            if cache_root.exists():
                result = subprocess.run(
                    ["du", "-sb", str(cache_root)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    size_bytes = int(result.stdout.split()[0])
                    hf_cache_size_gb = round(size_bytes / (1024**3), 1)
        except Exception as e:
            print(f"  Warning: Could not calculate HF cache size: {e}")

        template_vars = {
            "gpu_count": effective_gpu_count,
            "gpu_total_count": gpu_count,
            "gpu_names": gpu_names[0] if gpu_names else "unknown",
            "gpu_memory": gpu_memory,
            "gpu_memory_total": effective_gpu_count * gpu_memory,
            "compute_capability": compute_capability,
            "model_folder_size_gb": model_folder_size_gb,
            "hf_cache_size_gb": hf_cache_size_gb,
            "cuda_version": device.get("cuda", {}).get("version", "unknown"),
            "system_ram": device.get("system", {}).get("total_ram_gb", 0),
            "model_id": model_id,
            "max_position": max_position,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "estimated_params": f"{estimated_params:.1f}",
            "architecture": model_cfg.get("architectures", ["unknown"])[0]
            if model_cfg.get("architectures")
            else "unknown",
            "quantization": model_cfg.get("quantization_config", {}).get(
                "quant_method", "none"
            ),
            "model_dir": str(self.model_dir),
            "lm_start_dir": str(Path(__file__).parent),
            "os_platform": platform.platform(),
            "python_version": platform.python_version(),
            "cuda_driver": device.get("cuda", {}).get("driver_version", "unknown"),
            "opencode_bin": self.opencode_bin,
            "history_count": len(history),
            "experiment_history": self._build_experiment_history(history),
            "optimization_mode": self.optimization_mode,
            "target_context": target_context,
            "calc_max_num_seqs": calc_max_num_seqs,
            "calc_batched_tokens": calc_batched_tokens,
            "calc_gpu_util": calc_gpu_util,
            "memory_warning": False,
            "seq_warning": False,
        }

        planner_prompts = self.prompts.get("planner", {})

        sections = [
            planner_prompts.get("planner_system", ""),
            planner_prompts.get("experiment_naming", ""),
            planner_prompts.get("file_permissions", ""),
            planner_prompts.get("intelligent_planning", ""),
            template_vars["experiment_history"],
            planner_prompts.get("decision_framework", ""),
            planner_prompts.get("error_patterns", ""),
            planner_prompts.get("task_instructions", ""),
            planner_prompts.get("critical_rules", ""),
            planner_prompts.get("experiment_history_header", ""),
            planner_prompts.get("failed_run_detail", ""),
            planner_prompts.get("successful_run_detail", ""),
            planner_prompts.get("no_history", ""),
            planner_prompts.get("parent_linking", ""),
            planner_prompts.get("outputs", ""),
            planner_prompts.get("output_format", ""),
        ]

        prompt = "\n\n".join(section for section in sections if section)

        pm = PromptManager()
        return pm._render_template(prompt, template_vars)

    def _load_experiment_history_from_runs(self) -> List[Dict]:
        history = []
        if not self.runs_dir.exists():
            return history

        for run_dir in sorted(self.runs_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue

            result_file = run_dir / "result.json"
            config_file = run_dir / "config.json"
            benchmark_file = run_dir / "benchmark.json"

            if result_file.exists():
                try:
                    with open(result_file) as f:
                        result = json.load(f)

                    entry = {
                        "run_id": result.get(
                            "run_id", run_dir.name.replace("run_", "")
                        ),
                        "success": result.get("success", False),
                        "config": {},
                        "metrics": {},
                        "error": None,
                    }

                    if config_file.exists():
                        with open(config_file) as f:
                            entry["config"] = json.load(f)

                    if benchmark_file.exists() and entry["success"]:
                        with open(benchmark_file) as f:
                            benchmark = json.load(f)
                            if "summary" in benchmark:
                                entry["metrics"] = benchmark["summary"]

                    if not entry["success"]:
                        log_analysis = result.get("log_analysis", {})
                        if log_analysis.get("errors"):
                            entry["error"] = log_analysis["errors"][0]

                    history.append(entry)
                except Exception as e:
                    _console.print(f"  [yellow]⚠[/yellow]  Could not load history from {run_dir}: {e}")

        return history

    def _build_experiment_history(self, history: List[Dict]) -> str:
        if not history:
            return "=== EXPERIMENT HISTORY ===\nNo previous runs - this is the first experiment.\n"

        lines = ["=== EXPERIMENT HISTORY ===", ""]
        lines.append("PREVIOUS RUNS:")

        for entry in history:
            run_id = entry.get("run_id", "unknown")
            success = entry.get("success", False)
            config = entry.get("config", {})
            metrics = entry.get("metrics", {})
            error = entry.get("error")

            lines.append(f"\nRun: run_{run_id}")
            lines.append(f"Status: {'SUCCESS' if success else 'FAILED'}")
            lines.append(
                f"Config: max_model_len={config.get('max_model_len', '?')}, "
                f"TP={config.get('tensor_parallel_size', '?')}, "
                f"backend={config.get('attention_backend', 'AUTO-DETECT')}"
            )

            if success and metrics:
                throughput = metrics.get("avg_throughput") or metrics.get(
                    "overall_score"
                )
                lines.append(
                    f"Performance: {throughput} tok/s throughput"
                    if throughput
                    else "Performance: completed"
                )
            elif error:
                error_str = str(error)
                if len(error_str) > 100:
                    lines.append(f"Error: {error_str[:100]}...")
                else:
                    lines.append(f"Error: {error_str}")

            lines.append(
                f"Result: {'Working config found' if success else 'Failed - see error above'}"
            )

        lines.append("\n=== END HISTORY ===")
        return "\n".join(lines)

        lines = []

        if failed_runs:
            lines.append(f"FAILED RUNS (READ these to understand what went wrong):")
            lines.append(f"  Count: {len(failed_runs)}")
            lines.append("")

            for exp in failed_runs[-5:]:  # Show last 5 failed
                summary = exp.get("summary", {})
                error_type = exp.get("error_type", "unknown")
                run_id = exp.get("run_id", "unknown")
                lines.extend(
                    [
                        f"  Run {run_id}:",
                        f"    Config: {summary.get('one_line', 'no details')}",
                        f"    Error Type: {error_type}",
                        f"    Status: FAILED - Must read to fix",
                        f"    Files to read:",
                        f"      cat .runs/{run_id}/result.json",
                        f"      head -50 .runs/{run_id}/vllm_stderr.log",
                        f"      cat .runs/{run_id}/config.json",
                        "",
                    ]
                )

        if successful_runs:
            lines.append(f"SUCCESSFUL RUNS (Build upon these):")
            lines.append(f"  Count: {len(successful_runs)}")
            lines.append("")

            for exp in successful_runs[-3:]:  # Show last 3 successful
                summary = exp.get("summary", {})
                metrics = summary.get("metrics", {})
                ctx = exp.get("config", {}).get("max_model_len", 0)
                run_id = exp.get("run_id", "unknown")
                lines.extend(
                    [
                        f"  Run {run_id}:",
                        f"    Context: {ctx:,} tokens",
                        f"    Throughput: {metrics.get('avg_throughput', 0):.1f} tok/s",
                        f"    TTFT: {metrics.get('avg_ttft_ms', 0):.1f} ms",
                        f"    Config: {summary.get('one_line', 'no details')}",
                        f"    Status: WORKING - Can optimize further or increase context",
                        "",
                    ]
                )

        return "\n".join(lines)

    def execute_experiment(self, experiment: Dict) -> Dict[str, Any]:
        """
        Execute a single experiment.

        Creates run directory, spawns vLLM, runs benchmark.
        """
        run_name = experiment.get("name", f"run_{experiment['id']}")
        run_id = run_name
        config = experiment["config"]

        requested_port = config.get("port", 8000)
        if self._is_port_in_use(requested_port):
            port = self._find_free_port(start_port=8000, end_port=9000)
            _console.print(f"  [yellow]⚠[/yellow]  Port {requested_port} busy, switching to {port}")
        else:
            port = requested_port

        config["port"] = port

        _console.print(f"")
        _console.rule(f"[bold cyan]{run_id}[/bold cyan]", style="dim")
        _console.print(f"  [bold]Phase 1/3  ·  vLLM startup[/bold]  [dim](port {port})[/dim]")
        self._kill_existing_vllm(target_port=port)

        # Apply profile if specified
        profile = experiment.get("profile", "balanced")
        if profile in ["speed", "quality", "balanced"]:
            generator = ProfileConfigGenerator(self.model_dir)
            profiles = generator.generate_profiles(config)
            profile_map = {
                "speed": "performance",
                "quality": "quality",
                "balanced": "balanced",
            }
            config = profiles.get(profile_map[profile], config)

        run_dir = self.runs_dir / (
            run_id if str(run_id).startswith("run_") else f"run_{run_id}"
        )

        if run_dir.exists():
            suffix = uuid.uuid4().hex[:4]
            if str(run_id).startswith("run_"):
                run_id = f"{run_id}_{suffix}"
            else:
                run_id = f"run_{run_id}{suffix}"
            run_dir = self.runs_dir / run_id
            print(f"  ℹ  Run dir collision, using: {run_id}")

        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_file = run_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        # Save metadata
        meta_file = run_dir / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "name": run_name,
                    "hypothesis": experiment.get("hypothesis", ""),
                    "profile": profile,
                    "timestamp_started": datetime.now().isoformat(),
                    "venv_mode": experiment.get("venv_mode", "inherit"),
                    "insights_read": getattr(self, "_current_insights_read", ""),
                    "knowledge_update": getattr(self, "_current_knowledge_update", ""),
                    "previous_run": experiment.get("previous_run"),
                    "reasoning": experiment.get("reasoning", ""),
                    "prediction": experiment.get("prediction"),
                },
                f,
                indent=2,
            )

        # Build vLLM command
        venv_python = self.model_dir / ".venv" / "bin" / "python"
        vllm_cmd = self._build_vllm_command(config)

        try:
            stdout_log = open(run_dir / "vllm_stdout.log", "w")
            stderr_log = open(run_dir / "vllm_stderr.log", "w")

            # Get model's venv path to prevent inheriting wrong VIRTUAL_ENV
            model_venv = self.model_dir / ".venv"

            process = subprocess.Popen(
                vllm_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.model_dir,
                env=self.config.get_env_dict(
                    str(model_venv) if model_venv.exists() else None
                ),
                text=True,
                bufsize=1,
            )

            import threading

            backend_workers_seen = [0]  # mutable counter for thread-safe tracking

            def stream_output(pipe, log_file, prefix, printed_msgs):
                for line in iter(pipe.readline, ""):
                    if line:
                        log_file.write(line)
                        log_file.flush()
                        msg = line.strip()
                        if msg not in printed_msgs:
                            if "ERROR" in line or "CRITICAL" in line:
                                _console.print(f"  [red]⚠[/red]  {msg[:120]}")
                                printed_msgs.add(msg)
                            elif "AttentionBackend" in line:
                                # Count workers, print summary only on first occurrence
                                backend_workers_seen[0] += 1
                                raw = msg.split("AttentionBackend")[-1].strip().split()[0] if "AttentionBackend" in msg else "?"
                                # Strip enum prefix e.g. "Enum.FLASH_ATTN" -> "FLASH_ATTN"
                                backend = raw.split(".")[-1] if "." in raw else raw
                                if backend_workers_seen[0] == 1:
                                    _console.print(f"  [dim cyan]⚡  Attention: {backend}[/dim cyan]")
                                printed_msgs.add(msg)
                            elif "GPU KV cache" in line or "Maximum concurrency" in line:
                                # Extract just the meaningful part after the log prefix
                                import re as _re
                                clean = _re.sub(r'^.*?\] ', '', msg).strip()
                                _console.print(f"  [dim]⚡  {clean[:120]}[/dim]")
                                printed_msgs.add(msg)

            printed_messages = set()
            stdout_thread = threading.Thread(
                target=stream_output,
                args=(process.stdout, stdout_log, "vLLM", printed_messages),
            )
            stderr_thread = threading.Thread(
                target=stream_output,
                args=(process.stderr, stderr_log, "vLLM", printed_messages),
            )
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Wait for health check
            health_ok, elapsed_time = self._wait_for_health(port, process, timeout=1800)

            if not health_ok:
                _console.print(f"  [red]✗[/red]  vLLM failed to start")
                process.terminate()
                result = {
                    "run_id": run_id,
                    "success": False,
                    "error": "Health check timeout",
                    "error_type": "STARTUP_TIMEOUT",
                }
            else:
                _console.print(f"  [green]✓[/green]  vLLM healthy  [dim]({elapsed_time:.1f}s)[/dim]")
                _console.print(f"\n  [bold]Phase 2/3  ·  Benchmarking[/bold]")
                benchmark = self._run_benchmark(port, run_dir, config)

                bench = benchmark.get("summary", {})
                throughput = bench.get("avg_throughput", 0)
                ttft = bench.get("avg_ttft_ms", 0)
                success_rate = bench.get("success_rate", 0) * 100
                _console.print(
                    f"  [green]✓[/green]  {throughput:.0f} tok/s  |  TTFT {ttft:.0f} ms  |  Success {success_rate:.0f}%"
                )
                _console.print(f"  [dim]Stopping vLLM...[/dim]")
                process.terminate()
                process.wait(timeout=30)

                result = {
                    "run_id": run_id,
                    "success": True,
                    "benchmark": benchmark,
                    "config": config,
                }

        except Exception as e:
            result = {
                "run_id": run_id,
                "success": False,
                "error": str(e),
                "error_type": "EXECUTION_ERROR",
            }

        stdout_log = run_dir / "vllm_stdout.log"
        stderr_log = run_dir / "vllm_stderr.log"

        if stdout_log.exists() or stderr_log.exists():
            try:
                log_analysis = parse_vllm_logs(stdout_log, stderr_log)

                log_analysis_summary = {
                    k: v for k, v in log_analysis.items() if k != "errors"
                }

                if log_analysis["likely_success"] and result.get("success"):
                    result["success"] = True
                    result["log_analysis"] = log_analysis_summary
                elif not log_analysis["likely_success"] and result.get("success"):
                    result["success"] = False
                    result["error"] = (
                        "Log analysis indicates failure despite health check"
                    )
                    result["error_type"] = log_analysis.get(
                        "error_classification", {}
                    ).get("type", "UNKNOWN")
                    result["log_analysis"] = log_analysis_summary
                elif log_analysis["error_classification"]:
                    result["error_type"] = log_analysis["error_classification"].get(
                        "type", result.get("error_type", "UNKNOWN")
                    )
                    result["log_analysis"] = log_analysis_summary

            except Exception as e:
                print(f"Warning: Log parsing failed: {e}")

        # Save result
        result_file = run_dir / "result.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def _build_vllm_command(self, config: Dict) -> List[str]:
        venv_python = self.model_dir / ".venv" / "bin" / "python"

        # Build flag_config dynamically from config, excluding internal keys
        flag_config = {}
        internal_keys = {
            "_profile",
            "_profile_description",
            "compilation_config",
            "host",
            "port",
        }

        for key, value in config.items():
            if key.startswith("_") or key in internal_keys:
                continue
            flag_config[key] = value

        validated_args = validate_vllm_flags(flag_config, model_dir=str(self.model_dir))

        return [
            str(venv_python),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_id,
            "--host",
            config.get("host", "0.0.0.0"),
            "--port",
            str(config.get("port", 8000)),
        ] + validated_args

    def _wait_for_health(
        self, port: int, process: subprocess.Popen, timeout: int = 1800
    ) -> tuple[bool, float]:
        """Wait for vLLM health check, abort if process dies."""
        start = time.time()
        url = f"http://localhost:{port}/health"

        _console.print(f"  [dim]⏳  Waiting for vLLM...[/dim]")

        while time.time() - start < timeout:
            # Check if process is still running
            if process.poll() is not None:
                _console.print(f"  [red]✗[/red]  vLLM process died  [dim](exit {process.returncode})[/dim]")
                return False, time.time() - start

            try:
                response = requests.get(url, timeout=2)
                elapsed = time.time() - start
                # Caller prints the health-ok message; just return
                return True, elapsed
            except Exception:
                pass
            time.sleep(1)

        return False, time.time() - start

    def _run_benchmark(self, port: int, run_dir: Path, config: Dict) -> Dict[str, Any]:
        """Run all enabled benchmarks against running vLLM using BenchmarkRunner."""

        venv_python = self.model_dir / ".venv" / "bin" / "python"
        runner = BenchmarkRunner(
            model_id=self.model_id,
            vllm_port=port,
            venv_python=venv_python,
            verbose=True,
        )

        # Load benchmark registry from ExperimentAgent
        from lm_start.agents.experiment_agent import ExperimentAgent
        benchmark_registry = ExperimentAgent.BENCHMARK_REGISTRY

        # Load config overrides
        try:
            import yaml
            system_config_path = Path.home() / ".lm-start" / "config" / "system.yaml"
            if system_config_path.exists():
                with open(system_config_path) as f:
                    system_config = yaml.safe_load(f)
                benchmark_configs = system_config.get("experiment", {}).get("benchmarks", [])
                for bench_config in benchmark_configs:
                    if "name" in bench_config:
                        for reg_entry in benchmark_registry:
                            if reg_entry["name"] == bench_config["name"]:
                                reg_entry.update(bench_config)
                                break
        except Exception:
            pass

        results = {}
        all_passed = True

        for bench_spec in benchmark_registry:
            if not bench_spec.get("enabled", True):
                continue

            bench_name = bench_spec["name"]
            bench_method = bench_spec["method"]

            _console.print(f"  [dim]Running benchmark: {bench_name}[/dim]")

            try:
                if not hasattr(runner, bench_method):
                    continue

                method = getattr(runner, bench_method)

                if bench_name == "comprehensive":
                    result = method(config, quick_mode=True)
                    passed = result.get("summary", {}).get("overall_success", False)
                elif bench_name == "stress_test":
                    target_percent = bench_spec.get("target_percent", 0.75)
                    seq_headroom = bench_spec.get("seq_headroom", 2048)
                    result = method(config, target_percent=target_percent, seq_headroom=seq_headroom)
                    passed = not result.get("crashed", True)
                else:
                    result = method(config)
                    passed = result.get("success", False)

                results[bench_name] = result

                if not passed:
                    all_passed = False
                    _console.print(f"  [yellow]⚠[/yellow]  {bench_name} had failures")
                else:
                    _console.print(f"  [green]✓[/green]  {bench_name} complete")

            except Exception as e:
                _console.print(f"  [red]✗[/red]  {bench_name} failed: {e}")
                results[bench_name] = {"error": str(e), "crashed": True}
                all_passed = False

        aggregated = {
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "individual_results": results,
            "all_passed": all_passed,
            "benchmarks_run": list(results.keys()),
            "benchmarks_passed": [name for name, r in results.items()
                                 if not r.get("crashed") and not r.get("error")],
        }

        with open(run_dir / "benchmark.json", "w") as f:
            json.dump(aggregated, f, indent=2, default=str)

        if all_passed:
            _console.print(f"  [green]✓[/green]  All benchmarks complete")
        else:
            _console.print(f"  [yellow]⚠[/yellow]  Some benchmarks had failures")

        return aggregated

    def spawn_summarizer_agent(self, run_id: str, result: Dict) -> Dict[str, Any]:
        """Spawn Run Summarizer Agent via OpenCode CLI."""
        _console.print(f"\n  [bold]Phase 3/3  ·  Summarizing[/bold]")

        run_dir = self.runs_dir / (
            run_id if str(run_id).startswith("run_") else f"run_{run_id}"
        )

        # Build prompt
        prompt = self._build_summarizer_prompt(run_id, run_dir, result)

        # Save prompt
        prompt_file = self.opencode_dir / f"summarizer_prompt_{run_id}.txt"
        with open(prompt_file, "w") as f:
            f.write(prompt)

        # Add result file
        result_file = run_dir / "result.json"
        config_file = run_dir / "config.json"

        session_title = f"Summarizer-{run_id}-{uuid.uuid4().hex[:8]}"

        cmd = [
            self.opencode_bin,
            "run",
            "--model",
            "Grid/kimi-latest",
            "--agent",
            "build",
            "--dir",
            str(self.model_dir),
            "--file",
            str(prompt_file),
            "--title",
            session_title,
            "--format",
            "json",
            "summarize",
        ]

        if result_file.exists():
            cmd.extend(["--file", str(result_file)])
        if config_file.exists():
            cmd.extend(["--file", str(config_file)])

        try:
            sub_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                env=self.config.get_env_dict(str(self.model_dir / ".venv")),
            )

            time.sleep(2)
            session_id = self._find_session_by_title(session_title)
            if not session_id:
                session_id = self._extract_session_id(
                    sub_result.stdout + sub_result.stderr
                )
            if not session_id:
                session_id = self._get_latest_opencode_session("recovery")
            if session_id:
                self._save_session(
                    session_id,
                    "summarizer",
                    run_id=run_id,
                    phase="optimize",
                    details={
                        "success": result.get("success", False),
                        "error_type": result.get("error_type", "none"),
                        "config_max_len": result.get("config", {}).get(
                            "max_model_len", "unknown"
                        ),
                    },
                )

            self._archive_logs(run_id, run_dir)
            summary_file = run_dir / "summary.json"
            insights_file = self.model_dir / "Insights.md"

            files_created = []
            if summary_file.exists():
                files_created.append("summary.json")
            if insights_file.exists() and insights_file.stat().st_size > 200:
                files_created.append("Insights.md")

            if files_created:
                _console.print(f"  [green]✓[/green]  Summary saved: [dim]{', '.join(files_created)}[/dim]")
                return {"success": True, "files_created": files_created}
            else:
                _console.print(f"  [yellow]⚠[/yellow]  Summarizer ran but no files detected")
                _console.print(
                    f"  [dim]Output: {sub_result.stdout[:200] if sub_result.stdout else 'No output'}[/dim]"
                )
                return {"success": False, "error": "No summary files created"}

        except Exception as e:
            _console.print(f"  [yellow]⚠[/yellow]  Summarizer failed: {e}")
            return {"success": False, "error": str(e)}

    def _build_summarizer_prompt(self, run_id: str, run_dir: Path, result: Dict) -> str:
        summarizer_prompts = self.prompts.get("summarizer", {})
        prompt = summarizer_prompts.get("summarizer_system", "")

        prompt = prompt.replace("{{run_id}}", str(run_id))
        prompt = prompt.replace("{{run_dir}}", str(run_dir))
        prompt = prompt.replace("{{model_dir}}", str(self.model_dir))

        return prompt

    def _append_planner_insights_to_md(self, knowledge_update: str):
        insights_file = self.model_dir / "Insights.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        entry = f"\n## Planner Insight ({timestamp})\n\n{knowledge_update}\n"

        try:
            if insights_file.exists():
                with open(insights_file, "a") as f:
                    f.write(entry)
            else:
                with open(insights_file, "w") as f:
                    f.write(f"# Insights for {self.model_id}\n")
                    f.write(entry)
            pass  # insight appended silently
        except Exception as e:
            print(f"  Warning: Could not append to Insights.md: {e}")

    def _append_insights_read_to_md(self, insights_read: str):
        """Append planner's research documentation to Insights.md."""
        insights_file = self.model_dir / "Insights.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        entry = f"\n## Research Documentation ({timestamp})\n\n{insights_read}\n"

        try:
            if insights_file.exists():
                with open(insights_file, "a") as f:
                    f.write(entry)
            else:
                with open(insights_file, "w") as f:
                    f.write(f"# Insights for {self.model_id}\n")
                    f.write(entry)
            pass  # research doc appended silently
        except Exception as e:
            print(f"  Warning: Could not append to Insights.md: {e}")

    def _archive_logs(self, run_id: str, run_dir: Path):
        """Archive logs to .logs/ directory."""
        import tarfile

        log_dir = self.logs_dir / f"run_{run_id}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create tarball
        tar_path = log_dir / "logs.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for log_file in run_dir.glob("*.log"):
                tar.add(log_file, arcname=log_file.name)

        print(f"✓ Logs archived to {tar_path}")

    def _extract_json_from_output(self, output: str) -> Dict:
        """Extract JSON from opencode output."""
        import re

        debug_file = self.opencode_dir / "parser_debug.txt"
        debug_lines = []
        debug_lines.append(f"=== PARSER DEBUG {datetime.now().isoformat()} ===")
        debug_lines.append(f"Output length: {len(output)} chars")
        debug_lines.append(f"Output preview (first 500 chars): {output[:500]}")
        debug_lines.append("")

        code_block_pattern = r"```(?:json)?\s*\n(.*?)\n```"
        code_blocks = re.findall(code_block_pattern, output, re.DOTALL)
        debug_lines.append(f"Found {len(code_blocks)} code blocks")

        for i, block in enumerate(code_blocks):
            debug_lines.append(f"  Block {i}: {len(block)} chars")
            try:
                result = json.loads(block.strip())
                debug_lines.append(f"  Block {i}: SUCCESS")
                debug_lines.append(f"  Keys: {list(result.keys())}")
                with open(debug_file, "w") as f:
                    f.write("\n".join(debug_lines))
                return result
            except Exception as e:
                debug_lines.append(f"  Block {i}: FAILED - {e}")
                continue

        debug_lines.append("Code block parsing failed, trying brace matching...")
        brace_count = 0
        start_idx = -1
        matches_found = 0

        for i, char in enumerate(output):
            if char == "{":
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    matches_found += 1
                    try:
                        candidate = output[start_idx : i + 1]
                        result = json.loads(candidate)
                        debug_lines.append(
                            f"  Brace match {matches_found} at {start_idx}:{i}: SUCCESS"
                        )
                        debug_lines.append(f"  Keys: {list(result.keys())}")
                        with open(debug_file, "w") as f:
                            f.write("\n".join(debug_lines))
                        return result
                    except Exception as e:
                        debug_lines.append(
                            f"  Brace match {matches_found} at {start_idx}:{i}: FAILED - {e}"
                        )
                        continue

        debug_lines.append(
            f"All parsing failed. Total brace matches attempted: {matches_found}"
        )
        debug_lines.append("Returning empty plan.")

        with open(debug_file, "w") as f:
            f.write("\n".join(debug_lines))

        return {
            "experiments_queued": [],
            "optimization_complete": True,
            "reasoning": "Failed to parse planner output",
        }

    def run(self):
        """
        Main optimization loop.

        Sequential execution due to GPU constraint.
        """
        self.setup_directories()

        # Copy Insights.md template to model directory if it doesn't exist
        self._setup_insights_file()

        iteration = 0
        max_iterations = 20

        model_context = self.load_model_context()

        _console.print()
        _console.rule(f"[bold cyan]Agentic Optimizer[/bold cyan]  [dim]{self.model_id}[/dim]")
        _console.print(f"  Max iterations: {max_iterations}")
        _console.rule(style="dim")

        while iteration < max_iterations:
            iteration += 1
            _console.print()
            _console.rule(f"[dim]Iteration {iteration}/{max_iterations}[/dim]", style="dim")
            _console.print(f"  [cyan]🤖  Consulting planner...[/cyan]")

            # 1. Spawn Planner Agent
            plan = self.spawn_planner_agent()
            num_experiments = len(plan.get("experiments_queued", []))

            if plan.get("optimization_complete") and num_experiments == 0:
                if self.experiments_run < self.min_experiments and not self.force:
                    _console.print("  [yellow]⚠[/yellow]  Planner suggests completion, but minimum iterations not met")
                    _console.print("  [dim]→ Generating baseline experiment...[/dim]")
                    plan["experiments_queued"] = [
                        {
                            "id": f"baseline_{self.experiments_run + 1}",
                            "name": "baseline_max_context",
                            "priority": "high",
                            "hypothesis": "Test max context with aggressive batching",
                            "config": {
                                "max_model_len": model_context.get(
                                    "model_config", {}
                                ).get("max_position_embeddings", 32768),
                                "tensor_parallel_size": model_context.get(
                                    "device_config", {}
                                ).get("gpu_count", 8),
                                "max_num_batched_tokens": 65536,
                                "max_num_seqs": 256,
                                "gpu_memory_utilization": 0.95,
                            },
                            "venv_mode": "inherit",
                        }
                    ]
                    plan["optimization_complete"] = False
                    num_experiments = 1
                else:
                    _console.print(f"\n  [green]✓[/green]  Optimization complete!")
                    break

            experiments = plan["experiments_queued"]
            total_experiments = len(experiments)

            for idx, experiment in enumerate(experiments, 1):
                exp_name = experiment.get("name", experiment["id"])

                result = self.execute_experiment(experiment)
                self.experiments_run += 1

                # Spawn summarizer to update Insights.md
                run_id = experiment.get("name", f"run_{self.experiments_run}")
                self.spawn_summarizer_agent(run_id, result)

                success = result.get("success", False)
                error = result.get("error", "")

                if success:
                    _console.print(f"  [green]✓[/green]  {run_id}  passed")
                else:
                    err_short = str(error)[:80] + ("..." if len(str(error)) > 80 else "")
                    _console.print(f"  [red]✗[/red]  {run_id}  failed  [dim]—  {err_short}[/dim]")

            _console.rule(style="dim")

            # Loop continues - next iteration will spawn planner again with updated history

        # Final summary
        self._generate_final_report()

    def spawn_run_summarizer_agent(self, create_master: bool = True) -> Dict[str, Any]:
        """Spawn agent to summarize all runs and create summary.json for each."""
        print(f"\n{'━' * 60}")
        print("📊 RUN SUMMARIZER AGENT")
        print(f"   Processing all runs in: {self.runs_dir}")
        print(f"{'━' * 60}")

        run_summarizer_prompts = self.prompts.get("run_summarizer", {})
        prompt = run_summarizer_prompts.get("run_summarizer_system", "")

        prompt = prompt.replace("{{model_dir}}", str(self.model_dir))
        prompt = prompt.replace("{{model_id}}", self.model_id)

        prompt_file = self.opencode_dir / "run_summarizer_prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(prompt)

        cmd = [
            self.opencode_bin,
            "--session",
            f"run_summarizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "--instruction",
            str(prompt_file),
            "--cwd",
            str(self.model_dir),
        ]

        print(f"Spawning run summarizer...")
        print(f"Command: {' '.join(cmd[:5])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            output = result.stdout + "\n" + result.stderr
            session_id = self._extract_session_id(output)

            result_data = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:]
                if len(result.stdout) > 2000
                else result.stdout,
                "stderr": result.stderr[-2000:]
                if len(result.stderr) > 2000
                else result.stderr,
                "session_id": session_id,
            }

            if result_data["success"]:
                print(f"   ✓ Run summarizer completed")
                if session_id:
                    print(f"   Session: {session_id}")
            else:
                print(f"   ✗ Run summarizer failed (exit {result.returncode})")

            return result_data

        except subprocess.TimeoutExpired:
            print("   ✗ Run summarizer timed out after 10 minutes")
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            print(f"   ✗ Run summarizer error: {e}")
            return {"success": False, "error": str(e)}

    def summarize_all_runs(self) -> Dict[str, Any]:
        """Summarize all existing runs without running optimization."""
        self.setup_directories()

        run_dirs = list(self.runs_dir.glob("run_*"))
        if not run_dirs:
            print("No runs found to summarize")
            return {"success": False, "error": "No runs found", "runs_count": 0}

        print(f"Found {len(run_dirs)} run directories to process")
        return self.spawn_run_summarizer_agent(create_master=True)

    def _generate_final_report(self):
        """Generate final optimization report."""
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)

        # Find best config
        best_config = None
        best_run_id = None
        best_score = 0

        for result_file in self.runs_dir.glob("*/result.json"):
            run_dir = result_file.parent
            run_id = run_dir.name.replace("run_", "")

            with open(result_file) as f:
                result = json.load(f)

            if result.get("success"):
                # Score based on context length
                config = result.get("config", {})
                context = config.get("max_model_len", 0)

                if context > best_score:
                    best_score = context
                    best_config = result
                    best_run_id = run_id

        if best_config:
            print(f"\nBest Configuration:")
            print(f"  Run ID: {best_run_id}")
            print(f"  Max Context: {best_score:,} tokens")
            print(f"  Config: {best_config['config']}")

            # Save best config
            best_file = self.model_dir / "vllm_config_best.json"
            with open(best_file, "w") as f:
                json.dump(best_config, f, indent=2)

            print(f"\n✓ Saved to: {best_file}")
        else:
            print("\n✗ No successful configurations found")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="OpenCode Agentic Optimizer")
    parser.add_argument("model_dir", help="Model directory path")
    parser.add_argument(
        "--model-id", required=True, help="Model ID (e.g., Qwen/Qwen2.5-7B)"
    )
    parser.add_argument("--goal", default="maximize context", help="Optimization goal")

    args = parser.parse_args()

    orchestrator = OpenCodeAgenticOrchestrator(
        model_dir=args.model_dir,
        model_id=args.model_id,
        goal=args.goal,
    )

    orchestrator.run()


if __name__ == "__main__":
    main()
