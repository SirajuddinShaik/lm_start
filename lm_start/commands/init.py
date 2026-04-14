import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from lm_start import config
from lm_start import constants
from lm_start import opencode as oc
from lm_start import system as sys_detect

console = Console()


def ensure_config_dir():
    config.ensure_config_dir()
    return True


def install_opencode(version: str = "1.4.0") -> bool:
    if oc.is_installed():
        installed_version = oc.get_version()
        if installed_version == version:
            console.print(f"[green]OpenCode v{version} already installed[/green]")
            return True
        else:
            console.print(
                f"[yellow]Upgrading OpenCode from v{installed_version} to v{version}...[/yellow]"
            )
    else:
        console.print(f"[cyan]Installing OpenCode v{version}...[/cyan]")

    return oc.ensure_version(version)


def generate_full_system_config(hardware: dict) -> dict:
    gpu_count = hardware.get("gpu_count", 0)
    gpu_memory_gb = hardware.get("gpu_memory_gb", 0)
    gpu_name = hardware.get("gpu_name", "")
    cpu_cores = hardware.get("cpu_cores", 0)
    ram_gb = hardware.get("ram_gb", 0)
    python_version = hardware.get("python_version", "")
    cuda_available = hardware.get("cuda_available", False)
    cuda_version = hardware.get("cuda_version", "")

    cuda_home = "/usr/local/cuda"
    if cuda_version:
        major_minor = cuda_version.split(".")[:2]
        cuda_home = f"/usr/local/cuda-{'.'.join(major_minor)}"

    visible_devices = (
        ",".join(str(i) for i in range(gpu_count)) if gpu_count > 0 else ""
    )

    compute_capability = "7.0"
    if "H100" in gpu_name or "H200" in gpu_name:
        compute_capability = "9.0"
    elif "A100" in gpu_name:
        compute_capability = "8.0"
    elif "RTX 40" in gpu_name:
        compute_capability = "8.9"
    elif "RTX 30" in gpu_name:
        compute_capability = "8.6"

    system_config = {
        "system": {
            "name": "lm_start",
            "version": "2.0.0",
            "base_dir": "/data/models",
            "max_retries": 5,
            "phase_timeout": 1800,
            "experiment_timeout": 600,
        },
        "device": {
            "name": "auto-detected",
            "description": f"{gpu_count}x {gpu_name}"
            if gpu_count > 0
            else "CPU-only system",
            "gpu": {
                "count": gpu_count,
                "names": [gpu_name] if gpu_name else [],
                "memory_gb": int(gpu_memory_gb / gpu_count) if gpu_count > 0 else 0,
                "visible_devices": visible_devices,
                "compute_capability": compute_capability,
            },
            "cuda": {
                "version": cuda_version,
                "home": cuda_home,
                "driver_version": "",
            }
            if cuda_available
            else {
                "version": "",
                "home": "",
                "driver_version": "",
            },
            "system_ram_gb": int(ram_gb),
            "cpu_count": cpu_cores,
        },
        "paths": {
            "hf_home": "/data/.cache/huggingface",
            "models_base": "/data/models",
            "logs_base": "/data/.cache/lm_start/logs",
            "temp": "/tmp",
        },
        "environment": {
            "CUDA_HOME": cuda_home if cuda_available else "",
            "CUDA_VISIBLE_DEVICES": visible_devices if gpu_count > 0 else "",
            "LD_LIBRARY_PATH": f"{cuda_home}/lib64:/usr/lib/x86_64-linux-gnu"
            if cuda_available
            else "",
            "HF_HOME": "/data/.cache/huggingface",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TORCH_CUDA_MATMUL_PRECISION": "high",
            "PYTHONHASHSEED": "0",
            "PROMETHEUS_MULTIPROC_DIR": "/tmp/vllm_prometheus_multiproc",
        },
        "vllm_defaults": {
            "host": "0.0.0.0",
            "port": 8000,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
            "tensor_parallel_size": min(gpu_count, 8) if gpu_count > 0 else 1,
            "dtype": "auto",
            "trust_remote_code": True,
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "auto",
            "quantization": None,
            "max_num_seqs": None,
        },
        "pm2": {
            "log_date_format": "YYYY-MM-DD HH:mm:ss Z",
            "max_restarts": 10,
            "restart_delay": 5000,
            "kill_timeout": 60000,
            "listen_timeout": 300000,
            "min_uptime": "10s",
            "log_max_size": "100M",
        },
        "monitoring": {
            "health_check_interval": 5,
            "startup_timeout": 1800,
            "max_experiment_time": 3600,
            "health_endpoint": "/health",
        },
        "experiment": {
            "max_runs": 20,
            "context_targets": [8192, 16384, 32768, 65536, 131072],
            "test_port_range": [8001, 9000],
            "expectation": "Find the best context length and attention backend combination that achieves at least 32768 tokens context without OOM errors. Test FLASHINFER, FLASH_ATTN, FLASH_ATTN_VLLM_V1, XFORMERS, and TORCH_SDPA backends. Prioritize the backend with highest tokens/sec at 32768 context or above. Stop only when 32768 is achieved and stable.",
            "attention_backends": [
                "FLASHINFER",
                "FLASH_ATTN",
                "FLASH_ATTN_VLLM_V1",
                "XFORMERS",
                "TORCH_SDPA",
            ],
            "venv": {
                "base_path": ".venv",
                "per_run": True,
                "reuse_on_success": True,
                "cleanup_failed": False,
            },
        },
        "benchmark": {
            "enabled": True,
            "backend": "openai",
            "host": "localhost",
            "port": 8000,
            "endpoint": "/v1/completions",
            "dataset_name": "random",
            "random_input_len": 32000,
            "random_output_len": 1000,
            "num_prompts": 100,
            "request_rate": 32,
        },
        "smoke_test": {
            "enabled": True,
            "vllm_args": {
                "max_model_len": 1024,
                "tensor_parallel_size": 1,
                "dtype": "auto",
                "trust_remote_code": True,
                "enable_chunked_prefill": True,
                "enforce_eager": False,
            },
            "timeout_seconds": 1800,
            "poll_interval_seconds": 2,
            "max_retries": 3,
            "start_port": 8000,
            "end_port": 9000,
        },
        "llm": {
            "model": "Grid/kimi-latest",
            "timeout": 300,
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "phases": {
            "order": [
                "init",
                "fetch_info",
                "download",
                "venv",
                "generate_config",
                "optimize",
                "finalize",
            ],
            "init": {
                "description": "Initialize model directory and state",
                "required": True,
            },
            "fetch_info": {
                "description": "Fetch model metadata from HuggingFace",
                "required": True,
            },
            "download": {
                "description": "Download model weights",
                "required": False,
                "skippable": True,
            },
            "venv": {
                "description": "Create virtual environment and install dependencies",
                "required": True,
                "python_version": "3.12",
                "vllm_version": "latest",
            },
            "generate_config": {
                "description": "Generate PM2 and management scripts",
                "required": True,
            },
            "optimize": {
                "description": "Optimize vLLM configuration",
                "required": False,
                "modes": ["auto", "research"],
            },
            "finalize": {
                "description": "Create documentation and finalize",
                "required": True,
            },
        },
        "recovery": {
            "enabled": True,
            "auto_retry": True,
            "use_llm": True,
            "max_llm_attempts": 5,
            "strategies": [
                "reduce_context",
                "reduce_tensor_parallel",
                "enable_eager",
                "change_dtype",
                "adjust_memory_util",
            ],
        },
    }

    return system_config


def detect_and_save_hardware() -> dict:
    try:
        hardware = sys_detect.detect_hardware()
        full_config = generate_full_system_config(hardware)
        config.save_yaml(constants.SYSTEM_FILE, full_config)
        return hardware
    except Exception as e:
        console.print(f"[yellow]Warning: Hardware detection failed: {e}[/yellow]")
        minimal_config = generate_full_system_config(
            {
                "gpu_count": 0,
                "gpu_memory_gb": 0.0,
                "gpu_name": "",
                "cpu_cores": 0,
                "ram_gb": 0.0,
                "python_version": "",
                "cuda_available": False,
                "cuda_version": "",
            }
        )
        config.save_yaml(constants.SYSTEM_FILE, minimal_config)
        return {
            "gpu_count": 0,
            "gpu_memory_gb": 0.0,
            "gpu_name": "",
            "cpu_cores": 0,
            "ram_gb": 0.0,
            "python_version": "",
            "cuda_available": False,
            "cuda_version": "",
        }


def prompt_hf_token() -> bool:
    credentials = config.load_yaml(constants.CREDENTIALS_FILE)

    if credentials.get("HF_TOKEN"):
        console.print("[green]HuggingFace token already configured[/green]")
        return True

    if os.environ.get("HF_TOKEN"):
        console.print(
            "[green]HuggingFace token found in HF_TOKEN environment variable[/green]"
        )
        return True

    console.print(
        "\n[cyan]To enable model downloads from HuggingFace, you can set HF_TOKEN:[/cyan]"
    )
    console.print("  [dim]Option 1: Set HF_TOKEN environment variable[/dim]")
    console.print("  [dim]Option 2: Run: lm-start env set HF_TOKEN <your-token>[/dim]")

    return True


def display_system_summary(hardware: dict):
    table = Table(
        title="[bold cyan]System Hardware Summary[/bold cyan]", show_header=False
    )
    table.add_column("Property", style="cyan", width=25)
    table.add_column("Value", style="green")

    if hardware.get("gpu_count", 0) > 0:
        gpu_info = f"{hardware['gpu_count']}x {hardware.get('gpu_name', 'GPU')}"
        gpu_mem = f"{hardware.get('gpu_memory_gb', 0):.1f} GB"
        table.add_row("GPU", gpu_info)
        table.add_row("GPU Memory", gpu_mem)
    else:
        table.add_row("GPU", "[yellow]No GPU detected[/yellow]")

    table.add_row("CPU Cores", str(hardware.get("cpu_cores", 0)))
    table.add_row("RAM", f"{hardware.get('ram_gb', 0):.1f} GB")
    table.add_row("Python", hardware.get("python_version", "Unknown"))

    if hardware.get("cuda_available"):
        table.add_row("CUDA", hardware.get("cuda_version", "Available"))
    else:
        table.add_row("CUDA", "[yellow]Not available[/yellow]")

    console.print("\n")
    console.print(table)


def run_init():
    console.print(
        Panel.fit("[bold cyan]lm-start Initialization[/bold cyan]", border_style="cyan")
    )

    steps = [
        ("Creating config directory", lambda: ensure_config_dir()),
        ("Installing OpenCode v1.4.0", lambda: install_opencode("1.4.0")),
        ("Detecting hardware", lambda: detect_and_save_hardware()),
        ("Checking HuggingFace credentials", lambda: prompt_hf_token()),
    ]

    hardware = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for step_name, step_func in steps:
            task = progress.add_task(step_name, total=None)
            try:
                result = step_func()
                if step_name == "Detecting hardware":
                    hardware = result
                progress.update(task, completed=True)
            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"[yellow]Warning: {step_name} failed: {e}[/yellow]")

    display_system_summary(hardware)

    console.print("\n")
    console.print(
        Panel.fit(
            "[bold green]✓ Initialization complete![/bold green]\n\n"
            "Run [cyan]lm-start --help[/cyan] to see available commands.",
            border_style="green",
        )
    )


if __name__ == "__main__":
    run_init()
