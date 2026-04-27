"""
Setup phases for model deployment.
Matches setup_model.sh behavior exactly.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from rich.console import Console as _Console

_con = _Console()

from lm_start.state_manager import StateManager, Phase, PhaseStatus, PHASE_ORDER

# Import script modules for direct execution (pip package compatible)
from lm_start.scripts.fetch_model_info import fetch_model_info_func
from lm_start.scripts.download_model import download_model_func
from lm_start.scripts.extract_vllm_flags import extract_vllm_flags_func
from lm_start.utils.model_config_generator import generate_model_config

SCRIPT_DIR = Path(__file__).parent.parent / "scripts"


def get_credentials_env():
    try:
        credentials_file = Path.home() / ".lm-start" / "config" / "credentials.yaml"
        if credentials_file.exists():
            import yaml

            with open(credentials_file) as f:
                data = yaml.safe_load(f)
            return data.get("environment_variables", {})
    except Exception:
        pass
    return {}


def get_config_value(key: str = "", default: Any = None) -> Any:
    try:
        # Check user config first, then package config
        user_config = Path.home() / ".lm-start" / "config" / "system.yaml"
        package_config = SCRIPT_DIR / "config" / "system.yaml"
        system_config_path = user_config if user_config.exists() else package_config
        if system_config_path.exists():
            import yaml

            with open(system_config_path) as f:
                config = yaml.safe_load(f)
            if not key:
                return config
            keys = key.split(".")
            value = config
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default
            return value if value is not None else default
    except Exception:
        pass
    return default


class PhaseResult:
    def __init__(self, success: bool, message: str = "", data: Dict = None):
        self.success = success
        self.message = message
        self.data = data or {}


def run_script(
    script_path: Path,
    args: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict] = None,
    activate_venv: bool = True,
    stream_output: bool = False,
):
    cmd = (
        ["python3", str(script_path)] + args
        if str(script_path).endswith(".py")
        else [str(script_path)] + args
    )

    merged_env = os.environ.copy()
    merged_env.update(get_credentials_env())
    if env:
        merged_env.update(env)

    # FIX 1: Activate venv before running scripts
    if activate_venv and "VIRTUAL_ENV" not in merged_env:
        lmstart_venv = SCRIPT_DIR / ".venv"
        if lmstart_venv.exists():
            merged_env["PATH"] = (
                str(lmstart_venv / "bin") + ":" + merged_env.get("PATH", "")
            )
            merged_env["VIRTUAL_ENV"] = str(lmstart_venv)

    import subprocess

    if stream_output:
        process = subprocess.Popen(
            cmd,
            cwd=cwd or SCRIPT_DIR,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        stdout_lines = []
        for line in process.stdout:
            print(line, end="")
            stdout_lines.append(line)
        process.wait()
        result = subprocess.CompletedProcess(
            cmd, process.returncode, "".join(stdout_lines), ""
        )
        return result
    else:
        return subprocess.run(
            cmd, cwd=cwd or SCRIPT_DIR, capture_output=True, text=True, env=merged_env
        )


def run_agentic_recovery(model_dir: str, phase: str, error_msg: str) -> bool:
    import sys
    cmd = [sys.executable, "-m", "lm_start.scripts.opencode_phase_agent", model_dir, phase, error_msg]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, **get_credentials_env()},
    )
    
    return result.returncode == 0


def phase_init(
    model_id: str,
    model_dir: str,
    base_dir: str,
    runner: Optional[StateManager] = None,
    dry_run: bool = False,
):
    model_path = Path(model_dir)
    if runner and runner.get_phase_status(Phase.INIT) == PhaseStatus.COMPLETED:
        return PhaseResult(True, "Phase already completed")
    if runner:
        runner.start_phase(Phase.INIT)
    if dry_run:
        return PhaseResult(True, "Dry run - would create directory")
    try:
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "logs").mkdir(exist_ok=True)

        # Create detailed device_config.json using actual hardware detection
        from lm_start.core.hardware import HardwareDetector

        detector = HardwareDetector()
        hw_info = detector.detect()

        import psutil

        cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count() or 64
        total_ram_gb = psutil.virtual_memory().total / (1024**3)

        device_config = {
            "device_name": "Auto-Detected",
            "description": f"{hw_info.gpu_count}x {hw_info.gpus[0].name if hw_info.gpus else 'No GPU'}"
            if hw_info.gpus
            else "No GPU detected",
            "cuda": {
                "version": hw_info.cuda_version or "unknown",
                "home": "/usr/local/cuda" if hw_info.cuda_version else "",
            },
            "gpus": {
                "visible_devices": hw_info.visible_devices,
                "count": hw_info.gpu_count,
                "names": [gpu.name for gpu in hw_info.gpus],
                "memory_gb_per_gpu": round(hw_info.gpus[0].memory_total_gb, 1)
                if hw_info.gpus
                else 0,
                "total_memory_gb": round(hw_info.total_vram_gb, 1),
                "compute_capability": hw_info.gpus[0].compute_capability
                if hasattr(hw_info.gpus[0], "compute_capability")
                else "9.0"
                if hw_info.gpus
                else "",
            },
            "system": {
                "total_ram_gb": round(total_ram_gb, 1),
                "cpu_count": cpu_count,
            },
            "paths": {},
            "environment": {},
        }
        device_config_path = (
            model_path / ".llm-context" / "model-context" / "device_config.json"
        )
        device_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(device_config_path, "w") as f:
            json.dump(device_config, f, indent=2)

        if runner:
            runner.complete_phase(Phase.INIT)
        return PhaseResult(True, f"Initialized {model_dir}")
    except Exception as e:
        if runner:
            runner.fail_phase(Phase.INIT, str(e))
        return PhaseResult(False, str(e))


def phase_fetch_info(
    model_id: str,
    model_dir: str,
    runner: Optional[StateManager] = None,
    dry_run: bool = False,
):
    if runner and runner.get_phase_status(Phase.FETCH_INFO) == PhaseStatus.COMPLETED:
        return PhaseResult(True, "Phase already completed")
    if dry_run:
        return PhaseResult(True, "Dry run - would fetch model info")
    if runner:
        runner.start_phase(Phase.FETCH_INFO)
    hf_home = get_config_value(
        "paths.hf_home", os.path.expanduser("~/.cache/huggingface")
    )
    model_info_file = (
        Path(model_dir) / ".llm-context" / "model-context" / "model_info.json"
    )
    model_info_file.parent.mkdir(parents=True, exist_ok=True)
    result = fetch_model_info_func(
        hf_url=model_id,
        output_file=str(model_info_file),
        hf_home=hf_home,
    )

    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error")
        if runner:
            runner.fail_phase(Phase.FETCH_INFO, error_msg)
        return PhaseResult(False, f"Failed to fetch model info: {error_msg}")

    try:
        model_info = result.get("model_info", {})
        if not model_info.get("vllm_compatible", True):
            if runner:
                runner.complete_phase(Phase.FETCH_INFO)
                runner.skip_phase(Phase.DOWNLOAD, "GGUF model")
                runner.skip_phase(Phase.VENV, "GGUF model")
                runner.skip_phase(Phase.SMOKE_TEST, "GGUF model")
                runner.skip_phase(Phase.EXTRACT_VLLM_CONFIG, "GGUF model")
            return PhaseResult(True, "GGUF model - phases skipped")
    except:
        pass
    if runner:
        runner.complete_phase(Phase.FETCH_INFO)
    return PhaseResult(True, "Model info saved")


def phase_download(
    model_id: str,
    model_dir: str,
    runner: Optional[StateManager] = None,
    dry_run: bool = False,
):
    if runner and runner.get_phase_status(Phase.DOWNLOAD) == PhaseStatus.COMPLETED:
        return PhaseResult(True, "Phase already completed")
    if runner and runner.get_phase_status(Phase.DOWNLOAD) == PhaseStatus.SKIPPED:
        return PhaseResult(True, "Phase skipped")
    if dry_run:
        return PhaseResult(True, "Dry run - would download model")
    if runner:
        runner.start_phase(Phase.DOWNLOAD)
    hf_home = get_config_value(
        "paths.hf_home", os.path.expanduser("~/.cache/huggingface")
    )

    # Use direct function call instead of subprocess
    result = download_model_func(
        model_id=model_id,
        model_dir=model_dir,
        hf_home=hf_home,
    )

    if not result.get("success", False):
        error_msg = result.get("error", f"Download failed for {model_id}")
        _con.print("  [cyan]Agentic recovery for download...[/cyan]")
        if run_agentic_recovery(model_dir, "download", error_msg):
            _con.print("  [green]✓[/green]  Agent recovery succeeded — retrying download")
            retry_result = download_model_func(
                model_id=model_id,
                model_dir=model_dir,
                hf_home=hf_home,
            )
            if retry_result.get("success", False):
                if runner:
                    runner.complete_phase(Phase.DOWNLOAD)
                return PhaseResult(True, "Model downloaded after agent fix")
        if runner:
            runner.fail_phase(Phase.DOWNLOAD, error_msg)
        return PhaseResult(False, f"Download failed: {error_msg}")
    if runner:
        runner.complete_phase(Phase.DOWNLOAD)
    return PhaseResult(True, "Model downloaded successfully")


def phase_venv(
    model_dir: str,
    runner: Optional[StateManager] = None,
    dry_run: bool = False,
    no_venv: bool = False,
    no_install: bool = False,
    python_version: Optional[str] = None,
    vllm_version: str = "latest",
):
    if runner and runner.get_phase_status(Phase.VENV) == PhaseStatus.COMPLETED:
        return PhaseResult(True, "Phase already completed")
    if runner and runner.get_phase_status(Phase.VENV) == PhaseStatus.SKIPPED:
        return PhaseResult(True, "Phase skipped")
    if no_venv or no_install:
        if runner:
            runner.start_phase(Phase.VENV)
            runner.skip_phase(Phase.VENV, "Skipped by user request")
        return PhaseResult(True, "Venv skipped by user")
    venv_script = SCRIPT_DIR / "create_venv.sh"
    if not venv_script.exists():
        return PhaseResult(False, "create_venv.sh not found")
    if dry_run:
        return PhaseResult(True, "Dry run - would create venv")
    if runner:
        runner.start_phase(Phase.VENV)
    args = [model_dir]
    if python_version:
        args.append(python_version)
    if vllm_version != "latest":
        args.append(vllm_version)
    result = subprocess.run(
        ["bash", str(venv_script)] + args,
        capture_output=True,
        text=True,
        env={**os.environ, **get_credentials_env()},
    )
    if result.returncode == 0:
        if runner:
            runner.complete_phase(Phase.VENV)
        return PhaseResult(True, "Virtual environment created")
    else:
        error_msg = f"Virtual environment creation failed"
        _con.print("  [cyan]Agentic recovery for venv...[/cyan]")
        if run_agentic_recovery(model_dir, "venv", error_msg):
            _con.print("  [green]✓[/green]  Agent recovery succeeded — retrying venv creation")
            retry_result = subprocess.run(
                ["bash", str(venv_script)] + args,
                capture_output=True,
                text=True,
                env={**os.environ, **get_credentials_env()},
            )
            if retry_result.returncode == 0:
                if runner:
                    runner.complete_phase(Phase.VENV)
                return PhaseResult(True, "Virtual environment created after agent fix")
        if runner:
            runner.fail_phase(Phase.VENV, result.stderr)
        return PhaseResult(False, f"Failed to create venv: {result.stderr}")


def phase_smoke_test(
    model_dir: str,
    model_id: str,
    runner: Optional[StateManager] = None,
    dry_run: bool = False,
):
    if runner and runner.get_phase_status(Phase.SMOKE_TEST) == PhaseStatus.COMPLETED:
        return PhaseResult(True, "Phase already completed")
    if runner and runner.get_phase_status(Phase.SMOKE_TEST) == PhaseStatus.SKIPPED:
        return PhaseResult(True, "Phase skipped")
    venv_path = Path(model_dir) / ".venv"
    if not venv_path.exists():
        if runner:
            runner.start_phase(Phase.SMOKE_TEST)
            runner.skip_phase(Phase.SMOKE_TEST, "No virtual environment")
        return PhaseResult(True, "No venv - skipping smoke test")
    if dry_run:
        return PhaseResult(True, "Dry run - would run smoke test")
    if runner:
        runner.start_phase(Phase.SMOKE_TEST)

    import socket
    import requests

    model_path = Path(model_dir)
    vllm_bin = venv_path / "bin" / "vllm"
    smoke_log = model_path / "vllm_smoke_test.log"
    smoke_config = model_path / "smoke_test_config.yaml"
    system_config_path = SCRIPT_DIR / "config" / "system.yaml"

    # Create smoke_test_config.yaml from system.yaml (like shell)
    if not smoke_config.exists() and system_config_path.exists():
        import yaml

        system = yaml.safe_load(open(system_config_path))
        smoke = system.get("smoke_test", {})

        # Use actual GPU count so large models don't OOM with TP=1
        device_config_file = model_path / ".llm-context" / "model-context" / "device_config.json"
        tp_size = 1
        if device_config_file.exists():
            try:
                import json as _json
                dc = _json.load(open(device_config_file))
                tp_size = dc.get("gpus", {}).get("count", 1)
            except Exception:
                pass

        config = {
            "smoke_test": {
                "vllm_args": smoke.get(
                    "vllm_args",
                    {
                        "max_model_len": 1024,
                        "tensor_parallel_size": tp_size,
                        "dtype": "auto",
                        "trust_remote_code": True,
                        "enable_chunked_prefill": True,
                    },
                ),
                "timeout_seconds": smoke.get("timeout_seconds", 1800),
                "poll_interval_seconds": smoke.get("poll_interval_seconds", 2),
                "max_retries": smoke.get("max_retries", 3),
                "start_port": smoke.get("start_port", 29500),
            }
        }
        # Ensure tensor_parallel_size is set correctly even if vllm_args came from system.yaml
        config["smoke_test"]["vllm_args"]["tensor_parallel_size"] = tp_size
        yaml.dump(config, open(smoke_config, "w"), default_flow_style=False)
        _con.print(f"  [dim]Created smoke_test_config.yaml (TP={tp_size})[/dim]")

    model_info_file = model_path / ".llm-context" / "model-context" / "model_info.json"
    if not model_info_file.exists():
        _con.print("  [red]✗[/red]  model_info.json not found")
        if runner:
            runner.fail_phase(Phase.SMOKE_TEST, "model_info.json missing")
        return PhaseResult(
            False,
            "model_info.json not found - fetch_info phase may have failed",
        )

    with open(model_info_file) as f:
        model_data = json.load(f)
    actual_model_id = model_data.get("model_id", model_id)

    # Load smoke test settings
    if smoke_config.exists():
        import yaml

        smoke_settings = yaml.safe_load(open(smoke_config)).get("smoke_test", {})
    else:
        smoke_settings = {
            "timeout_seconds": 1800,
            "poll_interval_seconds": 2,
            "max_retries": 3,
            "start_port": 29500,
            "vllm_args": {},
        }

    timeout = smoke_settings.get("timeout_seconds", 1800)
    poll_interval = smoke_settings.get("poll_interval_seconds", 2)
    max_retries = smoke_settings.get("max_retries", 3)
    start_port = smoke_settings.get("start_port", 29500)
    vllm_args_dict = smoke_settings.get("vllm_args", {})

    # Convert vllm_args to command line args
    vllm_args_list = []
    for k, v in vllm_args_dict.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                vllm_args_list.append(flag)
            else:
                vllm_args_list.append(f"--no-{k.replace('_', '-')}")
        elif v == "" or v is None:
            vllm_args_list.append(flag)
        else:
            vllm_args_list.extend([flag, str(v)])

    device_config_file = (
        model_path / ".llm-context" / "model-context" / "device_config.json"
    )
    if device_config_file.exists():
        with open(device_config_file) as f:
            device_config = json.load(f)
        visible_devices = device_config.get("gpus", {}).get(
            "visible_devices", "0,1,2,3"
        )
    else:
        visible_devices = "0,1,2,3"

    # Find free port
    test_port = start_port
    for _ in range(100):  # Try 100 ports
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", test_port)) != 0:
                break
        test_port += 1

    _con.print(f"  [dim]Using port {test_port}[/dim]")
    _con.print(f"  [dim]Running smoke test (timeout: {timeout}s)...[/dim]")

    # Run smoke test
    env = {
        **os.environ,
        **get_credentials_env(),
        "CUDA_VISIBLE_DEVICES": visible_devices,
    }

    # First attempt
    healthy = False

    def run_one_test(port):
        """Run one smoke test attempt, reloading config fresh from disk."""
        import yaml as _yaml
        # Reload config each time so agent fixes are picked up
        if smoke_config.exists():
            _smoke_settings = _yaml.safe_load(open(smoke_config)).get("smoke_test", {})
        else:
            _smoke_settings = {}
        _vllm_args_dict = _smoke_settings.get("vllm_args", {})
        _vllm_args_list = []
        for k, v in _vllm_args_dict.items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    _vllm_args_list.append(flag)
                else:
                    _vllm_args_list.append(f"--no-{k.replace('_', '-')}")
            elif v == "" or v is None:
                _vllm_args_list.append(flag)
            else:
                _vllm_args_list.extend([flag, str(v)])

        nonlocal healthy
        cmd = (
            [str(vllm_bin), "serve", actual_model_id]
            + _vllm_args_list
            + ["--port", str(port)]
        )

        with open(smoke_log, "w") as log_f:
            process = subprocess.Popen(
                cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=model_dir
            )

        iterations = int(timeout / poll_interval)

        for i in range(iterations):
            time.sleep(poll_interval)

            if process.poll() is not None:
                _con.print(f"  [yellow]⚠[/yellow]  vLLM process died during startup")
                break

            try:
                elapsed = (i + 1) * poll_interval
                response = requests.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    _con.print(f"  [green]✓[/green]  vLLM healthy  [dim]({elapsed}s)[/dim]")
                    healthy = True
                    break
            except:
                pass

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except:
                process.kill()

        return healthy

    # First attempt
    healthy = run_one_test(test_port)

    max_agent_attempts = 5
    agent_attempt = 0

    while not healthy and agent_attempt < max_agent_attempts:
        error_log = smoke_log.read_text() if smoke_log.exists() else "No log file"
        agent_attempt += 1
        _con.print(
            f"  [yellow]\u26a0[/yellow]  Smoke test failed \u2014 agentic recovery [dim](attempt {agent_attempt}/{max_agent_attempts})[/dim]"
        )

        if run_agentic_recovery(model_dir, "smoke_test", error_log[:1000]):
            _con.print("  [green]\u2713[/green]  Agent recovery succeeded \u2014 retrying smoke test")
            test_port += 1
            healthy = run_one_test(test_port)
            if healthy:
                _con.print(
                    f"  [green]\u2713[/green]  Smoke test passed  [dim](after agent fix, attempt {agent_attempt})[/dim]"
                )
                if runner:
                    runner.complete_phase(Phase.SMOKE_TEST)
                return PhaseResult(
                    True,
                    f"vLLM smoke test passed after agent fix (attempt {agent_attempt})",
                )
            else:
                _con.print(
                    f"  [yellow]\u26a0[/yellow]  Still failing after agent recovery attempt {agent_attempt}"
                )
        else:
            _con.print("  [yellow]\u26a0[/yellow]  Agentic recovery could not fix the issue")
            break

    if not healthy:
        if runner:
            runner.fail_phase(Phase.SMOKE_TEST, "Max agent attempts reached")
        return PhaseResult(False, "vLLM smoke test failed - max agent attempts reached")

    _con.print("  [green]✓[/green]  Smoke test passed")
    if runner:
        runner.complete_phase(Phase.SMOKE_TEST)
    return PhaseResult(True, "vLLM smoke test passed")


def phase_extract_vllm_config(
    model_dir: str, runner: Optional[StateManager] = None, dry_run: bool = False
):
    if (
        runner
        and runner.get_phase_status(Phase.EXTRACT_VLLM_CONFIG) == PhaseStatus.COMPLETED
    ):
        return PhaseResult(True, "Phase already completed")
    if (
        runner
        and runner.get_phase_status(Phase.EXTRACT_VLLM_CONFIG) == PhaseStatus.SKIPPED
    ):
        return PhaseResult(True, "Phase skipped")
    if (
        runner
        and runner.get_phase_status(Phase.EXTRACT_VLLM_CONFIG) == PhaseStatus.FAILED
    ):
        _con.print("  [dim]Retrying previously failed extraction...[/dim]")
    venv_path = Path(model_dir) / ".venv"
    if not venv_path.exists():
        if runner:
            runner.start_phase(Phase.EXTRACT_VLLM_CONFIG)
            runner.skip_phase(Phase.EXTRACT_VLLM_CONFIG, "No virtual environment")
        return PhaseResult(True, "No venv - skipping extract")
    if dry_run:
        return PhaseResult(True, "Dry run - would extract config")
    if runner:
        runner.start_phase(Phase.EXTRACT_VLLM_CONFIG)

    # Use model's venv Python for extraction
    venv_python = Path(model_dir) / ".venv" / "bin" / "python"
    if not venv_python.exists():
        error_msg = f"venv Python not found: {venv_python}"
        _con.print(f"  [red]✗[/red]  {error_msg}")
        if runner:
            runner.fail_phase(Phase.EXTRACT_VLLM_CONFIG, error_msg)
        return PhaseResult(False, error_msg)

    hf_home = get_config_value("paths.hf_home", "/data/.cache/huggingface")

    # Run extraction in model's venv using subprocess
    import subprocess
    import json

    extract_script = SCRIPT_DIR / "extract_vllm_config.py"
    result = subprocess.run(
        [
            str(venv_python),
            str(extract_script),
            model_dir,
            "--max-model-len",
            "1024",
            "--json",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HF_HOME": hf_home},
    )

    if result.returncode != 0:
        error_msg = result.stderr or "Extraction failed"
        _con.print(f"  [red]✗[/red]  vLLM config extraction failed: {error_msg[:200]}")
        if runner:
            runner.fail_phase(Phase.EXTRACT_VLLM_CONFIG, error_msg)
        return PhaseResult(False, f"Extraction failed: {error_msg[:200]}")

    try:
        result_data = json.loads(result.stdout)
    except json.JSONDecodeError:
        error_msg = "Failed to parse extraction output"
        _con.print(f"  [red]✗[/red]  {error_msg}")
        if runner:
            runner.fail_phase(Phase.EXTRACT_VLLM_CONFIG, error_msg)
        return PhaseResult(False, error_msg)

    if not result_data.get("success", False):
        error_msg = result_data.get("error", "Unknown error")
        _con.print(f"  [red]✗[/red]  vLLM config extraction failed: {error_msg[:200]}")
        _con.print("  [cyan]Agentic recovery will attempt to fix this...[/cyan]")
        if runner:
            runner.fail_phase(Phase.EXTRACT_VLLM_CONFIG, error_msg)
        return PhaseResult(False, f"Extraction failed: {error_msg[:200]}")

    # Also extract vLLM flags (like shell script)
    # Must use model's venv Python, not lm-start's
    _con.print("  [dim]Extracting vLLM flags...[/dim]")

    model_venv_python = Path(model_dir) / ".venv" / "bin" / "python"
    extract_script = Path(__file__).parent.parent / "scripts" / "extract_vllm_flags.py"

    if model_venv_python.exists() and extract_script.exists():
        result = subprocess.run(
            [
                str(model_venv_python),
                str(extract_script),
                model_dir,
                "--profile",
                "balanced",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            _con.print("  [green]✓[/green]  vLLM flags extracted")
        else:
            error_msg = result.stderr[:100] if result.stderr else "Unknown error"
            _con.print(f"  [yellow]⚠[/yellow]  Flag extraction: {error_msg}")
    else:
        _con.print("  [yellow]⚠[/yellow]  Cannot extract flags: venv or script not found")

    # Extract comprehensive vLLM flags as YAML
    _con.print("  [dim]Extracting comprehensive vLLM flags...[/dim]")
    try:
        from lm_start.scripts.extract_vllm_flags import extract_vllm_flags_func
        result = extract_vllm_flags_func(model_dir)
        if result.get("success"):
            _con.print(f"  [green]\u2713[/green]  Extracted {result.get('flags_count', 0)} flags to vllm_flags.yaml")
        else:
            _con.print(f"  [yellow]\u26a0[/yellow]  Flag extraction: {result.get('error', 'unknown error')}")
    except Exception as e:
        _con.print(f"  [yellow]\u26a0[/yellow]  Failed to extract vLLM flags: {e}")

    sys.path.insert(0, str(SCRIPT_DIR))
    from utils.model_utils import get_model_cache_path

    model_dir_path = Path(model_dir)
    hub_configs_dir = model_dir_path / ".llm-context" / "hub_configs"
    hub_configs_dir.mkdir(parents=True, exist_ok=True)

    model_info_file = (
        model_dir_path / ".llm-context" / "model-context" / "model_info.json"
    )
    model_id_from_info = None
    if model_info_file.exists():
        with open(model_info_file) as f:
            model_info = json.load(f)
        model_id_from_info = model_info.get("model_id")

    cache_path = get_model_cache_path(
        model_id_from_info or model_dir_path.name, hf_home
    )

    all_files_to_copy = {}

    if cache_path and cache_path.exists():
        snapshots_dir = cache_path.parent
        if snapshots_dir.exists() and snapshots_dir.name == "snapshots":
            for snapshot_dir in snapshots_dir.iterdir():
                if snapshot_dir.is_dir():
                    for f in snapshot_dir.iterdir():
                        if f.is_file() and f.name not in all_files_to_copy:
                            all_files_to_copy[f.name] = f

    files_to_copy_to_hub = [
        f
        for name, f in all_files_to_copy.items()
        if "readme" in name.lower()
        or name.lower().endswith(".txt")
        or name
        in [
            "config.json",
            "tokenizer_config.json",
            "generation_config.json",
            "processor_config.json",
            "chat_template.jinja",
            ".gitattributes",
        ]
    ]

    for source_file in files_to_copy_to_hub:
        import shutil

        dest_file = hub_configs_dir / source_file.name
        shutil.copy2(source_file, dest_file)

    if files_to_copy_to_hub:
        file_list = ", ".join([f.name for f in files_to_copy_to_hub[:5]])
        if len(files_to_copy_to_hub) > 5:
            file_list += f", +{len(files_to_copy_to_hub) - 5} more"
        _con.print(f"  [dim]hub_configs: {len(files_to_copy_to_hub)} files ({file_list})[/dim]")

    if runner:
        runner.complete_phase(Phase.EXTRACT_VLLM_CONFIG)
    return PhaseResult(True, "vLLM config extracted")


def phase_generate_config(
    model_dir: str, runner: Optional[StateManager] = None, dry_run: bool = False
):
    if (
        runner
        and runner.get_phase_status(Phase.GENERATE_CONFIG) == PhaseStatus.COMPLETED
    ):
        return PhaseResult(True, "Phase already completed")
    if dry_run:
        return PhaseResult(True, "Dry run - would generate config")
    if runner:
        runner.start_phase(Phase.GENERATE_CONFIG)

    from lm_start.scripts.generate_pm2_config import generate_pm2_config_func
    from lm_start.scripts.generate_model_sh import generate_model_sh_func

    # Generate PM2 config
    result = generate_pm2_config_func(model_dir=model_dir)
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error")
        _con.print(f"  [yellow]⚠[/yellow]  PM2 config failed: {error_msg[:100]}")
        _con.print("  [dim]Continuing anyway...[/dim]")

    # Generate model.sh script
    result = generate_model_sh_func(model_dir=model_dir)
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error")
        _con.print(f"  [yellow]⚠[/yellow]  model.sh failed: {error_msg[:100]}")
        _con.print("  [dim]Continuing anyway...[/dim]")

    # Make model.sh executable
    model_sh = Path(model_dir) / "model.sh"
    if model_sh.exists():
        model_sh.chmod(0o755)

    if runner:
        runner.complete_phase(Phase.GENERATE_CONFIG)
    return PhaseResult(True, "Configs generated")


def phase_optimize(
    model_dir: str,
    model_id: str,
    runner: Optional[StateManager] = None,
    dry_run: bool = False,
    force: bool = False,
    agentic: bool = True,
):
    if runner and runner.get_phase_status(Phase.OPTIMIZE) == PhaseStatus.COMPLETED:
        return PhaseResult(True, "Phase already completed")
    if (
        runner
        and runner.get_phase_status(Phase.OPTIMIZE) == PhaseStatus.SKIPPED
        and not force
    ):
        return PhaseResult(True, "Phase skipped")
    if dry_run:
        return PhaseResult(True, "Dry run - would optimize")
    if runner:
        runner.start_phase(Phase.OPTIMIZE)

    from lm_start.scripts.optimizer import optimize_vllm_config

    max_attempts = 3
    attempt = 0
    last_error = None

    _con.print()
    _con.rule("[bold]Optimization Phase[/bold]")
    _con.print()

    while attempt < max_attempts:
        attempt += 1
        _con.print(f"  [dim]Attempt {attempt}/{max_attempts}[/dim]")

        result = optimize_vllm_config(
            model_dir=model_dir,
            min_context=8192,
            max_iterations=3,
            force=force,
            verbose=True,
            agentic=agentic,
        )

        if result.get("success", False):
            _con.print("  [green]✓[/green]  Optimization successful")
            _con.rule(style="dim")
            if runner:
                runner.complete_phase(Phase.OPTIMIZE)
            return PhaseResult(True, "Configuration optimized")

        _con.print(f"  [yellow]⚠[/yellow]  Attempt {attempt} failed")

        last_error = result.get("error", "Unknown error")
        error_preview = str(last_error)[-500:] if last_error else "No error output"
        _con.print(f"  [dim]Error:[/dim]")
        for line in error_preview.split("\n")[-5:]:
            if line.strip():
                _con.print(f"  [dim]  {line[:80]}{'...' if len(line) > 80 else ''}[/dim]")

        if attempt < max_attempts:
            _con.print("  [cyan]Agentic recovery...[/cyan]")
            if run_agentic_recovery(model_dir, "optimize", str(last_error)[:1000]):
                _con.print("  [green]✓[/green]  Recovery succeeded, retrying...")
            else:
                _con.print("  [red]✗[/red]  Recovery failed")
                break
        else:
            _con.print(f"  [yellow]⚠[/yellow]  Max attempts ({max_attempts}) reached")

    _con.rule(style="dim")
    if runner:
        runner.complete_phase(Phase.OPTIMIZE)
    return PhaseResult(True, "Optimization completed with warnings")


def phase_finalize(
    model_dir: str,
    model_id: str,
    model_name: str,
    runner: Optional[StateManager] = None,
    dry_run: bool = False,
):
    if runner and runner.get_phase_status(Phase.FINALIZE) == PhaseStatus.COMPLETED:
        return PhaseResult(True, "Phase already completed")
    if dry_run:
        return PhaseResult(True, "Dry run - would finalize")
    if runner:
        runner.start_phase(Phase.FINALIZE)

    model_dir_path = Path(model_dir)

    sys.path.insert(0, str(SCRIPT_DIR))
    from utils.model_utils import get_model_cache_path

    hf_home = get_config_value("paths.hf_home", "/data/.cache/huggingface")
    cache_path = get_model_cache_path(model_id, hf_home)

    model_context_dir = model_dir_path / ".llm-context" / "model-context"
    model_context_dir.mkdir(parents=True, exist_ok=True)

    if cache_path and cache_path.exists():
        all_files = list(cache_path.iterdir())

        readme_files = [
            f for f in all_files if f.is_file() and "readme" in f.name.lower()
        ]
        for readme_file in readme_files:
            import shutil

            dest_file = model_dir_path / readme_file.name
            shutil.copy2(readme_file, dest_file)
            _con.print(f"  [dim]Copied {readme_file.name} from HF cache[/dim]")
    else:
        readme_path = model_dir_path / "README.md"
        readme_path.write_text(
            f"# {model_name}\n\nModel: {model_id}\nDirectory: {model_dir}\n"
        )

    # Create .gitignore
    gitignore_content = """.venv/
logs/
*.log
__pycache__/
*.pyc
.DS_Store
.vllm_entrypoints_cache/
"""

    gitignore_path = Path(model_dir) / ".gitignore"
    with open(gitignore_path, "w") as f:
        f.write(gitignore_content)

    deploy_script = SCRIPT_DIR / "deploy.py"
    if deploy_script.exists():
        import shutil

        shutil.copy2(deploy_script, Path(model_dir) / "deploy.py")
        (Path(model_dir) / "deploy.py").chmod(0o755)
        _con.print("  [dim]Copied deploy.py[/dim]")

    if runner:
        runner.complete_phase(Phase.FINALIZE)
    return PhaseResult(True, "Setup finalized")
