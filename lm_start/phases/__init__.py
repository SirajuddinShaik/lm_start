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

_package_dir = Path(__file__).parent.parent.parent
if str(_package_dir) not in sys.path:
    sys.path.insert(0, str(_package_dir))

from state_manager import StateManager, Phase, PhaseStatus, PHASE_ORDER

SCRIPT_DIR = Path(__file__).parent.parent.parent


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
        system_config_path = SCRIPT_DIR / "config" / "system.yaml"
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
    agent_script = SCRIPT_DIR / "opencode_phase_agent.py"
    if not agent_script.exists():
        return False
    result = subprocess.run(
        ["python3", str(agent_script), model_dir, phase, error_msg],
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

        # Create detailed device_config.json like shell script
        system_config = get_config_value("", {})
        device = system_config.get("device", {})
        device_config = {
            "device_name": device.get("name", "unknown"),
            "description": device.get("description", ""),
            "cuda": {
                "version": device.get("cuda", {}).get("version", "12.9"),
                "home": device.get("cuda", {}).get("home", "/usr/local/cuda-12.9"),
            },
            "gpus": {
                "visible_devices": device.get("gpu", {}).get("visible_devices", "0"),
                "count": device.get("gpu", {}).get("count", 1),
                "names": device.get("gpu", {}).get("names", []),
                "memory_gb_per_gpu": device.get("gpu", {}).get("memory_gb", 80),
                "total_memory_gb": device.get("gpu", {}).get("count", 1)
                * device.get("gpu", {}).get("memory_gb", 80),
                "compute_capability": device.get("gpu", {}).get(
                    "compute_capability", "9.0"
                ),
            },
            "system": {
                "total_ram_gb": device.get("system_ram_gb", 512),
                "cpu_count": device.get("cpu_count", 64),
            },
            "paths": system_config.get("paths", {}),
            "environment": system_config.get("environment", {}),
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
    fetch_script = SCRIPT_DIR / "fetch_model_info.py"
    if not fetch_script.exists():
        return PhaseResult(False, "fetch_model_info.py not found")
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
    result = run_script(
        fetch_script, [model_id, "-o", str(model_info_file), "--hf-home", hf_home]
    )
    if result.returncode != 0:
        if runner:
            runner.fail_phase(Phase.FETCH_INFO, result.stderr)
        return PhaseResult(False, f"Failed to fetch model info: {result.stderr}")
    try:
        with open(model_info_file) as f:
            model_info = json.load(f)
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
    download_script = SCRIPT_DIR / "download_model.py"
    if not download_script.exists():
        return PhaseResult(False, "download_model.py not found")
    if dry_run:
        return PhaseResult(True, "Dry run - would download model")
    if runner:
        runner.start_phase(Phase.DOWNLOAD)
    hf_home = get_config_value(
        "paths.hf_home", os.path.expanduser("~/.cache/huggingface")
    )
    args = [model_id, "--model-dir", model_dir, "--hf-home", hf_home]
    if (Path(model_dir) / ".resume_download").exists():
        args.append("--resume")
    result = run_script(download_script, args)
    if result.returncode != 0:
        error_msg = f"Download failed for {model_id}"
        print("[INFO] Attempting agentic recovery for download...")
        if run_agentic_recovery(model_dir, "download", error_msg):
            print("[INFO] OpenCode agent recovery succeeded, retrying download...")
            result = run_script(download_script, args)
            if result.returncode == 0:
                if runner:
                    runner.complete_phase(Phase.DOWNLOAD)
                return PhaseResult(True, "Model downloaded after agent fix")
        if runner:
            runner.fail_phase(Phase.DOWNLOAD, result.stderr)
        return PhaseResult(False, f"Download failed: {result.stderr}")
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
        print("[INFO] Attempting agentic recovery for venv...")
        if run_agentic_recovery(model_dir, "venv", error_msg):
            print("[INFO] OpenCode agent recovery succeeded, retrying venv creation...")
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
        config = {
            "smoke_test": {
                "vllm_args": smoke.get(
                    "vllm_args",
                    {
                        "max_model_len": 1024,
                        "tensor_parallel_size": 1,
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
        yaml.dump(config, open(smoke_config, "w"), default_flow_style=False)
        print("[INFO] Created smoke_test_config.yaml")

    model_info_file = model_path / ".llm-context" / "model-context" / "model_info.json"
    if not model_info_file.exists():
        print("[ERROR] model_info.json not found in .llm-context/model-context/")
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

    print(f"[INFO] Using test port: {test_port}")
    print(f"[INFO] Running vLLM serve smoke test (timeout: {timeout}s)...")

    # Run smoke test
    env = {
        **os.environ,
        **get_credentials_env(),
        "CUDA_VISIBLE_DEVICES": visible_devices,
    }

    # First attempt
    healthy = False

    def run_one_test(port):
        """Run one smoke test attempt."""
        nonlocal healthy
        cmd = (
            [str(vllm_bin), "serve", actual_model_id]
            + vllm_args_list
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
                print("[WARN] vLLM process died during startup")
                break

            try:
                elapsed = (i + 1) * poll_interval
                response = requests.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    print(f"[OK] vLLM healthy after {elapsed}s")
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
        print(
            f"[INFO] Smoke test failed - calling agentic recovery (attempt {agent_attempt}/{max_agent_attempts})..."
        )

        if run_agentic_recovery(model_dir, "smoke_test", error_log[:1000]):
            print(
                "[INFO] OpenCode agent recovery succeeded, retrying with fixed config..."
            )
            test_port += 1
            healthy = run_one_test(test_port)
            if healthy:
                print(
                    f"[OK] vLLM smoke test passed after agent fix (attempt {agent_attempt})!"
                )
                if runner:
                    runner.complete_phase(Phase.SMOKE_TEST)
                return PhaseResult(
                    True,
                    f"vLLM smoke test passed after agent fix (attempt {agent_attempt})",
                )
            else:
                print(
                    f"[WARN] Smoke test still failed after agent recovery attempt {agent_attempt}"
                )
        else:
            print("[WARN] Agentic recovery could not fix the issue")
            break

    if not healthy:
        if runner:
            runner.fail_phase(Phase.SMOKE_TEST, "Max agent attempts reached")
        return PhaseResult(False, "vLLM smoke test failed - max agent attempts reached")

    print("[OK] vLLM smoke test passed!")
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
        print("[INFO] Retrying previously failed extraction...")
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

    extract_script = SCRIPT_DIR / "extract_vllm_config.py"
    if not extract_script.exists():
        return PhaseResult(False, "extract_vllm_config.py not found")
    python_exec = venv_path / "bin" / "python"
    hf_home = get_config_value("paths.hf_home", "/data/.cache/huggingface")
    result = subprocess.run(
        [str(python_exec), str(extract_script), model_dir, "--max-model-len", "1024"],
        capture_output=True,
        text=True,
        env={**os.environ, **get_credentials_env(), "HF_HOME": hf_home},
    )

    if result.returncode != 0:
        print(f"[ERROR] vLLM config extraction failed: {result.stderr[:200]}...")
        print("[INFO] Agentic recovery will attempt to fix this...")
        if runner:
            runner.fail_phase(Phase.EXTRACT_VLLM_CONFIG, result.stderr)
        return PhaseResult(False, f"Extraction failed: {result.stderr[:200]}")

    # FIX 3: Also extract vLLM flags (like shell script)
    flags_script = SCRIPT_DIR / "extract_vllm_flags.py"
    if flags_script.exists():
        print("[INFO] Extracting available vLLM flags...")
        flags_result = subprocess.run(
            [str(python_exec), str(flags_script), model_dir, "--profile", "balanced"],
            capture_output=True,
            text=True,
            env={**os.environ, **get_credentials_env()},
        )
        if flags_result.returncode != 0:
            print(f"[WARN] vLLM flag extraction failed: {flags_result.stderr[:100]}")
        else:
            print("[OK] vLLM flags extracted successfully")

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
        print(f"[OK] hub_configs: {len(files_to_copy_to_hub)} files ({file_list})")

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

    # Generate PM2 config
    pm2_script = SCRIPT_DIR / "generate_pm2_config.py"
    if pm2_script.exists():
        result = run_script(pm2_script, [model_dir])
        # FIX 4: Continue on error (like shell script)
        if result.returncode != 0:
            print(f"[WARN] Failed to generate PM2 config: {result.stderr[:100]}")
            print("[INFO] Continuing anyway...")

    # Generate model.sh script
    model_sh_script = SCRIPT_DIR / "generate_model_sh.py"
    if model_sh_script.exists():
        result = run_script(model_sh_script, [model_dir])
        if result.returncode != 0:
            print(f"[WARN] Failed to generate model.sh: {result.stderr[:100]}")
            print("[INFO] Continuing anyway...")

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

    optimize_script = SCRIPT_DIR / "optimize_vllm_config.py"
    if not optimize_script.exists():
        return PhaseResult(False, "optimize_vllm_config.py not found")

    args = ["--model-dir", model_dir, "--verbose"]
    if agentic:
        args.append("--agentic")

    max_attempts = 3
    attempt = 0
    result = None

    print("\n" + "━" * 60)
    print("🔬 OPTIMIZATION PHASE")
    print("━" * 60)

    while attempt < max_attempts:
        attempt += 1
        print(f"\n┌─ Optimization Attempt {attempt}/{max_attempts}")
        print("│")

        result = run_script(optimize_script, args)

        if result.returncode == 0:
            print("│")
            print("│  ✅ Optimization successful!")
            print("│")
            print("└─ Complete")
            print("\n" + "━" * 60)
            if runner:
                runner.complete_phase(Phase.OPTIMIZE)
            return PhaseResult(True, "Configuration optimized")

        print(f"│")
        print(f"│  ⚠️  Attempt {attempt} failed")

        if attempt < max_attempts:
            print("│")
            print("│  🔄 Running agentic recovery...")
            if run_agentic_recovery(model_dir, "optimize", result.stderr[:1000]):
                print("│  ✅ Recovery succeeded, retrying...")
            else:
                print("│  ❌ Recovery failed, stopping retries")
                print("│")
                print("└─ Aborted")
                break
        else:
            print(f"│")
            print(f"│  ⚠️  Max attempts ({max_attempts}) reached")
            print("│")
            print("└─ Continuing with warnings")

    print("\n" + "━" * 60)
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
            print(f"[OK] Copied {readme_file.name} from HF cache to model root")
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
        print("[OK] Copied deploy.py")

    if runner:
        runner.complete_phase(Phase.FINALIZE)
    return PhaseResult(True, "Setup finalized")
