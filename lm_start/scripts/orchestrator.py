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

from lm_start.utils.model_utils import get_model_cache_path, read_model_config
from lm_start.utils.theoretical_calculator import HardwareConfig, ModelConfig
from lm_start.utils.profile_config_generator import ProfileConfigGenerator, ProfileType
from lm_start.utils.benchmark_runner import BenchmarkRunner
from lm_start.utils.prompt_manager import PromptManager
from lm_start.utils.log_parser import parse_vllm_logs
from lm_start.vllm_flag_validator import validate_vllm_flags
from lm_start.core.config import get_config


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
        self.opencode_bin = "/home/ubuntu/.opencode/bin/opencode"

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

        match = re.search(r"ses_[a-zA-Z0-9]+", output)
        return match.group(0) if match else None

    def _get_latest_opencode_session(self, title_hint: str = None) -> Optional[str]:
        try:
            result = subprocess.run(
                [self.opencode_bin, "session", "list"],
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

        print(f"[SESSION] Saved: {session_id} ({agent_type})")
        if run_id:
            print(f"          Run ID: {run_id}")
        if phase:
            print(f"          Phase: {phase}")
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
            print(f"[WARN] Port {target_port} is already in use")
            proc = self._get_process_using_port(target_port)
            if proc:
                print(
                    f"[INFO] Port {target_port} is used by PID {proc.pid} ({proc.name()})"
                )
                pids_to_kill.add(proc.pid)
            else:
                print(f"[WARN] Could not identify process using port {target_port}")
        else:
            print(f"[DEBUG] Port {target_port} is free, no processes to kill")
            return killed

        for pid in pids_to_kill:
            try:
                proc = psutil.Process(pid)
                print(f"[INFO] Terminating process {pid} ({proc.name()})")
                proc.terminate()
                gone, alive = psutil.wait_procs([proc], timeout=10)
                if proc in alive:
                    print(f"[WARN] Process {pid} did not terminate, forcing kill")
                    proc.kill()
                    proc.wait(timeout=5)
                killed.append(pid)
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                print(f"[WARN] Error killing process {pid}: {e}")

        if killed:
            print(f"[OK] Killed {len(killed)} process(es): {killed}")
            time.sleep(3)

            # Verify port is freed
            if target_port:
                if self._is_port_in_use(target_port):
                    print(f"[ERROR] Port {target_port} is still in use!")
                else:
                    print(f"[OK] Port {target_port} is now free")

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
                print(
                    f"[WARN] Low GPU memory: {min_free:.1f}GB free (need {required_gb}GB)"
                )
                return False

            print(f"[OK] GPU memory check passed: {min_free:.1f}GB free")
            return True
        except Exception as e:
            print(f"[WARN] Could not check GPU memory: {e}")
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
        prompts_dir = Path(__file__).parent / "prompts"
        prompts = {}

        planner_file = prompts_dir / "opencode_planner.yaml"
        if planner_file.exists():
            with open(planner_file) as f:
                prompts["planner"] = yaml.safe_load(f)

        summarizer_file = prompts_dir / "opencode_summarizer.yaml"
        if summarizer_file.exists():
            with open(summarizer_file) as f:
                prompts["summarizer"] = yaml.safe_load(f)

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

    def spawn_planner_agent(self) -> Dict[str, Any]:
        """
        Spawn Master Planner Agent via OpenCode CLI.

        Returns experiment queue from planner.
        """
        print("\n" + "=" * 60)
        print("SPAWNING MASTER PLANNER AGENT")
        print("=" * 60)

        context = self.load_model_context()
        experiment_history = self._load_experiment_history_from_runs()
        prompt = self._build_planner_prompt(context, experiment_history)

        # Save prompt to file for opencode
        prompt_file = self.opencode_dir / "planner_prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(prompt)

        # Spawn opencode session in model directory
        cmd = [
            self.opencode_bin,
            "run",
            "--model",
            "Grid/kimi-latest",
            "--dir",
            str(self.model_dir),
            "planner",
            "--file",
            str(prompt_file),
        ]

        # Attach flag knowledge base files
        flag_kb = Path(__file__).parent / "configs" / "flag_knowledge_base.yaml"
        flag_rules = Path(__file__).parent / "configs" / "flag_rules.yaml"
        if flag_kb.exists():
            cmd.extend(["--file", str(flag_kb)])
        if flag_rules.exists():
            cmd.extend(["--file", str(flag_rules)])

        print(f"Running: planner agent with knowledge base...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=self.config.get_env_dict(),
            )

            debug_file = self.opencode_dir / "planner_output_debug.txt"
            with open(debug_file, "w") as f:
                f.write(f"STDOUT:\n{result.stdout}\n\n")
                f.write(f"STDERR:\n{result.stderr}\n\n")
                f.write(f"Return code: {result.returncode}\n")

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

            print(
                f"✓ Planner returned {len(plan.get('experiments_queued', []))} experiments"
            )
            if plan.get("reasoning"):
                print(f"  Reasoning: {plan['reasoning']}")
            return plan

        except subprocess.TimeoutExpired:
            print("✗ Planner timed out")
            return {"experiments_queued": [], "optimization_complete": True}
        except Exception as e:
            print(f"✗ Planner failed: {e}")
            return {"experiments_queued": [], "optimization_complete": True}

    def _build_planner_prompt(self, context: Dict, history: List[Dict]) -> str:
        """Build dynamic planner prompt from context using external prompt files."""
        device = context.get("device_config", {})
        model = context.get("model_info", {})
        model_cfg = context.get("model_config", {})

        # Extract device details
        gpu_info = device.get("gpus", {})
        gpu_count = gpu_info.get("count", 0)
        gpu_names = gpu_info.get("names", ["unknown"])
        gpu_memory = gpu_info.get("memory_gb_per_gpu", 0)
        compute_capability = gpu_info.get("compute_capability", "unknown")

        # Extract model details - fallback to model_info.json if HF config fails
        model_id = model.get("model_id", "unknown")

        # Try HF config first (handle nested text_config for models like gemma-4)
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

        # Fallback to model_info.json for unsupported architectures (like gemma4)
        if not max_position:
            max_position = model.get("context_length", 0)
        if not num_layers:
            num_layers = model_cfg.get("num_layers", 0)  # Some models store it here
        if not hidden_size:
            hidden_size = model_cfg.get("d_model", 0)  # Alternate field name
        if not num_attention_heads:
            num_attention_heads = model_cfg.get("num_heads", 0)  # Alternate field name

        # Estimate parameters
        estimated_params = (
            (num_layers * hidden_size * hidden_size * 12) / 1e9
            if num_layers and hidden_size
            else 0
        )

        # Build context variables for template substitution
        target_context = max_position if max_position else 32768
        calc_max_num_seqs = min(256, max(64, target_context // 256))
        calc_batched_tokens = min(65536, target_context * 2)
        calc_gpu_util = 0.90

        template_vars = {
            "gpu_count": gpu_count,
            "gpu_names": gpu_names[0] if gpu_names else "unknown",
            "gpu_memory": gpu_memory,
            "gpu_memory_total": gpu_count * gpu_memory,
            "compute_capability": compute_capability,
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
        }

        planner_prompts = self.prompts.get("planner", {})

        sections = [
            planner_prompts.get("planner_system", ""),
            "{{experiment_history}}",
            planner_prompts.get("decision_framework", ""),
            planner_prompts.get("error_patterns", ""),
            planner_prompts.get("task_instructions", ""),
            planner_prompts.get("critical_rules", ""),
            planner_prompts.get("output_format", ""),
        ]

        prompt = "\n\n".join(section for section in sections if section)

        # Simple template substitution
        for key, value in template_vars.items():
            placeholder = f"{{{{{key}}}}}"
            if isinstance(value, (list, dict)):
                value = json.dumps(value, indent=2)
            prompt = prompt.replace(placeholder, str(value))

        return prompt

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
                    print(f"[WARN] Could not load history from {run_dir}: {e}")

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
                lines.append(
                    f"Error: {error[:100]}..."
                    if len(str(error)) > 100
                    else f"Error: {error}"
                )

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
        run_id = experiment["id"]
        run_name = experiment.get("name", f"run_{run_id}")
        config = experiment["config"]

        requested_port = config.get("port", 8000)
        if self._is_port_in_use(requested_port):
            print(
                f"[WARN] Requested port {requested_port} is in use, finding alternative..."
            )
            port = self._find_free_port(start_port=8000, end_port=9000)
            print(f"[INFO] Using alternative port: {port}")
        else:
            port = requested_port

        config["port"] = port

        print(f"[DEBUG] execute_experiment started for {run_id}")
        print(f"\n[STEP 1/4] Preparing experiment environment...")
        print(f"           Port {port}: checking if in use...")
        print(f"[DEBUG] Killing existing vLLM processes on port {port}...")
        self._kill_existing_vllm(target_port=port)
        print(f"[DEBUG] Existing processes killed")

        print(f"[STEP 2/4] Testing vLLM health check...")

        print(f"\n" + "=" * 60)
        print(f"EXECUTING EXPERIMENT: {run_name} (ID: {run_id})")
        print("=" * 60)
        print(f"[DEBUG] Using profile: {experiment.get('profile', 'balanced')}")

        # Apply profile if specified
        profile = experiment.get("profile", "balanced")
        if profile in ["speed", "quality", "balanced"]:
            print(f"Applying profile: {profile}")
            generator = ProfileConfigGenerator(self.model_dir)
            profiles = generator.generate_profiles(config)
            profile_map = {
                "speed": "performance",
                "quality": "quality",
                "balanced": "balanced",
            }
            config = profiles.get(profile_map[profile], config)

        if str(run_id).startswith("run_"):
            run_dir = self.runs_dir / str(run_id)
        else:
            run_dir = self.runs_dir / f"run_{run_id}"

        if run_dir.exists():
            suffix = uuid.uuid4().hex[:4]
            if str(run_id).startswith("run_"):
                run_id = f"{run_id}_{suffix}"
            else:
                run_id = f"run_{run_id}_{suffix}"
            run_dir = self.runs_dir / run_id
            print(f"[INFO] Run ID collision detected, using: {run_id}")

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
                },
                f,
                indent=2,
            )

        # Build vLLM command
        venv_python = self.model_dir / ".venv" / "bin" / "python"
        vllm_cmd = self._build_vllm_command(config)

        print(f"│  │  🟢 Starting vLLM for {run_name} (waiting for health check)...")

        try:
            print(f"[DEBUG] Spawning vLLM process on port {port}...")
            print(f"│  │  🚀 vLLM starting...")

            stdout_log = open(run_dir / "vllm_stdout.log", "w")
            stderr_log = open(run_dir / "vllm_stderr.log", "w")

            process = subprocess.Popen(
                vllm_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.model_dir,
                env=self.config.get_env_dict(),
                text=True,
                bufsize=1,
            )

            print(f"[DEBUG] vLLM process started with PID: {process.pid}")
            print(f"│  │  📋 Streaming vLLM output...")

            import threading

            def stream_output(pipe, log_file, prefix):
                for line in iter(pipe.readline, ""):
                    if line:
                        log_file.write(line)
                        log_file.flush()
                        if "ERROR" in line or "error" in line.lower():
                            print(f"│  │  ⚠️  {prefix}: {line.strip()[:80]}")
                        elif (
                            "GPU KV cache" in line
                            or "Maximum concurrency" in line
                            or "Using" in line
                        ):
                            print(f"│  │  ✨ {prefix}: {line.strip()[:80]}")

            stdout_thread = threading.Thread(
                target=stream_output, args=(process.stdout, stdout_log, "vLLM")
            )
            stderr_thread = threading.Thread(
                target=stream_output, args=(process.stderr, stderr_log, "vLLM")
            )
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Wait for health check
            print(f"│  │  ⏳ Waiting for health check...")
            health_ok = self._wait_for_health(port, process, timeout=1800)

            if not health_ok:
                print(f"│  │  ❌ Health check failed")
                process.terminate()
                result = {
                    "run_id": run_id,
                    "success": False,
                    "error": "Health check timeout",
                    "error_type": "STARTUP_TIMEOUT",
                }
            else:
                print(f"│  │  ✅ Health check successful!")
                print(f"│  │")
                print(f"│  │  🔥 Starting benchmark...")
                benchmark = self._run_benchmark(port, run_dir, config)

                bench = benchmark.get("summary", {})
                throughput = bench.get("avg_throughput", 0)
                success_rate = bench.get("success_rate", 0) * 100
                print(f"│  │")
                print(
                    f"│  │  📊 Benchmark complete: {throughput:.1f} tok/s, {success_rate:.0f}% success"
                )
                print(f"│  │")
                print(f"│  │  🛑 Stopping vLLM...")
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

                if log_analysis["likely_success"] and result.get("success"):
                    result["success"] = True
                    result["log_analysis"] = log_analysis
                elif not log_analysis["likely_success"] and result.get("success"):
                    result["success"] = False
                    result["error"] = (
                        "Log analysis indicates failure despite health check"
                    )
                    result["error_type"] = log_analysis.get(
                        "error_classification", {}
                    ).get("type", "UNKNOWN")
                    result["log_analysis"] = log_analysis
                elif log_analysis["error_classification"]:
                    result["error_type"] = log_analysis["error_classification"].get(
                        "type", result.get("error_type", "UNKNOWN")
                    )
                    result["log_analysis"] = log_analysis

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
    ) -> bool:
        """Wait for vLLM health check, abort if process dies."""
        start = time.time()
        url = f"http://localhost:{port}/health"

        print(f"[DEBUG] _wait_for_health starting for port {port}")
        print(f"Waiting for vLLM health check at {url} (timeout: {timeout}s)...")

        check_count = 0
        while time.time() - start < timeout:
            check_count += 1
            if check_count % 12 == 0:
                elapsed = time.time() - start
                print(
                    f"[DEBUG] Still waiting for health check... {elapsed:.0f}s elapsed"
                )

            # Check if process is still running
            if process.poll() is not None:
                # Process died - read logs
                print(f"✗ vLLM process died (exit code: {process.returncode})")
                print(f"[DEBUG] Process exited at {time.time() - start:.1f}s")
                return False

            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    elapsed = time.time() - start
                    print(f"✓ vLLM healthy after {elapsed:.1f}s")
                    print(f"[DEBUG] Health check passed on attempt {check_count}")
                    return True
            except Exception as e:
                if check_count == 1:
                    print(
                        f"[DEBUG] First health check attempt failed (expected): {type(e).__name__}"
                    )
            time.sleep(5)

        print(f"✗ Health check timeout after {timeout}s")
        print(f"[DEBUG] Timeout after {check_count} attempts")
        return False

    def _run_benchmark(self, port: int, run_dir: Path, config: Dict) -> Dict[str, Any]:
        """Run benchmark against running vLLM using BenchmarkRunner."""
        print("Running benchmark...")

        venv_python = self.model_dir / ".venv" / "bin" / "python"
        runner = BenchmarkRunner(
            model_id=self.model_id,
            vllm_port=port,
            venv_python=venv_python,
            verbose=True,
        )

        try:
            result = runner.run_comprehensive_benchmark(config, quick_mode=True)

            with open(run_dir / "benchmark.json", "w") as f:
                json.dump(result, f, indent=2)

            if result.get("summary", {}).get("overall_success", False):
                print(f"✓ Benchmark completed")
            else:
                print(f"⚠ Benchmark had failures")

            return result

        except Exception as e:
            print(f"✗ Benchmark failed: {e}")
            return {"error": str(e)}

    def spawn_summarizer_agent(self, run_id: str, result: Dict) -> Dict[str, Any]:
        """
        Spawn Run Summarizer Agent via OpenCode CLI.

        Compresses logs and updates master plan.
        """
        print(f"\n" + "=" * 60)
        print(f"SPAWNING SUMMARIZER FOR RUN {run_id}")
        print("=" * 60)

        run_dir = self.runs_dir / f"run_{run_id}"

        # Build prompt
        prompt = self._build_summarizer_prompt(run_id, run_dir, result)

        # Save prompt
        prompt_file = self.opencode_dir / f"summarizer_prompt_{run_id}.txt"
        with open(prompt_file, "w") as f:
            f.write(prompt)

        # Add result file
        result_file = run_dir / "result.json"
        config_file = run_dir / "config.json"

        cmd = [
            self.opencode_bin,
            "run",
            "--model",
            "Grid/kimi-latest",
            "--dir",
            str(self.model_dir),
            "--file",
            str(prompt_file),
        ]

        if result_file.exists():
            cmd.extend(["--file", str(result_file)])
        if config_file.exists():
            cmd.extend(["--file", str(config_file)])

        log_analysis_file = run_dir / "log_analysis.json"
        if "log_analysis" in result:
            log_analysis_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_analysis_file, "w") as f:
                json.dump(result["log_analysis"], f, indent=2)
            cmd.extend(["--file", str(log_analysis_file)])

        cmd.append("summarize")

        print(f"Running: {' '.join(cmd[:8])}...")

        try:
            sub_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env=self.config.get_env_dict(),
            )

            session_id = self._extract_session_id(sub_result.stdout + sub_result.stderr)
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

            return {"success": True}

        except Exception as e:
            print(f"✗ Summarizer failed: {e}")
            return {"success": False, "error": str(e)}

    def _build_summarizer_prompt(self, run_id: str, run_dir: Path, result: Dict) -> str:
        """Build summarizer prompt using external prompt file."""
        success = result.get("success", False)
        error = result.get("error", "")
        error_type = result.get("error_type", "")
        config = result.get("config", {})

        template_vars = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "model_dir": str(self.model_dir),
            "success_status": "YES" if success else "NO",
            "success_bool": str(success).lower(),
            "max_model_len": config.get("max_model_len", "unknown"),
            "profile": config.get("_profile", "unknown"),
            "error_status": error_type if error else "None",
            "error_type": error_type,
        }

        summarizer_prompts = self.prompts.get("summarizer", {})
        prompt = summarizer_prompts.get("summarizer_system", "")

        for key, value in template_vars.items():
            placeholder = f"{{{{{key}}}}}"
            prompt = prompt.replace(placeholder, str(value))

        return prompt

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
        print("\n" + "=" * 60)
        print("OPENCODE AGENTIC OPTIMIZER")
        print(f"Model: {self.model_id}")
        print(f"Directory: {self.model_dir}")
        print("=" * 60)

        self.setup_directories()

        # Copy Insights.md template to model directory if it doesn't exist
        self._setup_insights_file()

        iteration = 0
        max_iterations = 20

        model_context = self.load_model_context()

        print(f"\n{'━' * 60}")
        print("🧪 AGENTIC OPTIMIZER")
        print(f"   Model: {self.model_id}")
        print(f"   Max Iterations: {max_iterations}")
        print(f"{'━' * 60}")

        while iteration < max_iterations:
            iteration += 1
            print(f"\n┌─ Iteration {iteration}/{max_iterations}")
            print("│")
            print("│  🤖 Consulting Planner Agent...")

            # 1. Spawn Planner Agent
            plan = self.spawn_planner_agent()
            num_experiments = len(plan.get("experiments_queued", []))

            if plan.get("optimization_complete") and num_experiments == 0:
                if self.experiments_run < self.min_experiments and not self.force:
                    print("│  ⚠️  Planner suggests completion, but minimum not met")
                    print("│  📝 Generating baseline experiment...")
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
                    print("│")
                    print("│  ✅ Optimization complete!")
                    print("│")
                    print("└─ Done")
                    break

            experiments = plan["experiments_queued"]
            total_experiments = len(experiments)

            for idx, experiment in enumerate(experiments, 1):
                exp_name = experiment.get("name", experiment["id"])
                exp_config = experiment.get("config", {})

                print(f"│  ┌─ Experiment {idx}/{total_experiments}: {exp_name}")
                print(f"│  │  🚀 Starting...")

                result = self.execute_experiment(experiment)
                self.experiments_run += 1

                success = result.get("success", False)
                error = result.get("error", "")

                if success:
                    bench = result.get("benchmark", {})
                    summary = bench.get("summary", {})
                    throughput = summary.get("avg_throughput", 0)
                    success_rate = summary.get("success_rate", 0) * 100

                    print(
                        f"│  │  ✅ Healthy | Throughput: {throughput:.1f} tok/s | Success: {success_rate:.0f}%"
                    )
                    print(f"│  └─ ✅ PASSED")
                else:
                    print(
                        f"│  │  ❌ Error: {error[:40]}{'...' if len(error) > 40 else ''}"
                    )
                    print(f"│  └─ ❌ FAILED")

            print("│")
            print("└─ Iteration complete")
            print(f"{'━' * 60}")

            # After running experiments, exit the loop
            break

        # Final summary
        self._generate_final_report()

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
