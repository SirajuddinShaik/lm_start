"""Doctor command - Run system diagnostics to verify setup."""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lm_start.constants import CREDENTIALS_FILE
from lm_start.system import detect_hardware

console = Console()


def check_opencode_installation() -> Tuple[bool, str]:
    """Check if OpenCode is installed and get its version."""
    opencode_path = Path.home() / ".lm-start" / "opencode" / "bin" / "opencode"

    if not opencode_path.exists():
        return False, f"OpenCode not found at {opencode_path}"

    try:
        result = subprocess.run(
            [str(opencode_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = result.stdout.strip() or result.stderr.strip()

        # Check if version matches 1.4.0
        if "1.4.0" in version:
            return True, f"OpenCode {version} installed"
        else:
            return True, f"OpenCode installed: {version} (expected 1.4.0)"
    except (
        subprocess.SubprocessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        return False, f"OpenCode found but failed to get version: {e}"


def check_credentials_file() -> Tuple[bool, str]:
    """Check if credentials file exists."""
    if CREDENTIALS_FILE.exists():
        return True, f"Credentials file found at {CREDENTIALS_FILE}"
    else:
        return False, f"Credentials file not found at {CREDENTIALS_FILE}"


def check_python_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Check if required Python dependencies are installed."""
    required = {
        "click": "click",
        "typer": "typer",
        "pyyaml": "yaml",
        "psutil": "psutil",
        "rich": "rich",
    }

    results = {}
    for pkg, import_name in required.items():
        try:
            __import__(import_name)
            results[pkg] = (True, f"{pkg} installed")
        except ImportError:
            results[pkg] = (False, f"{pkg} not installed")

    return results


def check_nvidia_smi() -> Tuple[bool, str]:
    """Check if nvidia-smi is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Get GPU info
            gpu_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            gpu_name = (
                gpu_result.stdout.strip() if gpu_result.returncode == 0 else "Unknown"
            )
            return True, f"nvidia-smi available ({gpu_name})"
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False, "nvidia-smi not available"


def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information using detect_hardware."""
    try:
        return detect_hardware()
    except Exception as e:
        return {
            "error": str(e),
            "gpu_count": 0,
            "gpu_memory_gb": 0.0,
            "gpu_name": "",
            "cpu_cores": 0,
            "ram_gb": 0.0,
            "python_version": "",
            "cuda_available": False,
            "cuda_version": "",
        }


def format_check(passed: bool, message: str) -> str:
    """Format a check result with color."""
    if passed:
        return f"[green]✓[/green] {message}"
    else:
        return f"[red]✗[/red] {message}"


def get_fix_suggestion(check_name: str, passed: bool) -> str:
    """Get suggestion for fixing an issue."""
    suggestions = {
        "opencode": "Run: mkdir -p ~/.lm-start/opencode/bin && curl -sL https://github.com/ohmycode-ai/opencode/releases/download/v1.4.0/opencode-linux-x64 -o ~/.lm-start/opencode/bin/opencode && chmod +x ~/.lm-start/opencode/bin/opencode",
        "credentials": f"Run: mkdir -p {CREDENTIALS_FILE.parent} && touch {CREDENTIALS_FILE}",
        "click": "Run: pip install click",
        "typer": "Run: pip install typer",
        "pyyaml": "Run: pip install pyyaml",
        "psutil": "Run: pip install psutil",
        "rich": "Run: pip install rich",
        "nvidia-smi": "Install NVIDIA drivers and CUDA toolkit",
    }

    if not passed and check_name in suggestions:
        return f"  [dim]Hint:[/dim] {suggestions[check_name]}"
    return ""


def run_doctor():
    """Run system diagnostics and display results."""
    console.print(
        Panel.fit(
            "[bold blue]🔍 lm-start Doctor - System Diagnostics[/bold blue]",
            border_style="blue",
        )
    )
    console.print()

    all_passed = True

    # 1. OpenCode Installation Check
    console.print("[bold]1. OpenCode Installation[/bold]")
    opencode_ok, opencode_msg = check_opencode_installation()
    console.print(f"  {format_check(opencode_ok, opencode_msg)}")
    if not opencode_ok:
        console.print(get_fix_suggestion("opencode", opencode_ok))
        all_passed = False
    console.print()

    # 2. Hardware Detection
    console.print("[bold]2. Hardware Detection[/bold]")
    hw = get_hardware_info()

    if "error" in hw:
        console.print(f"  [red]✗[/red] Failed to detect hardware: {hw['error']}")
    else:
        console.print(f"  CPU Cores: [cyan]{hw.get('cpu_cores', 'N/A')}[/cyan]")
        console.print(f"  RAM: [cyan]{hw.get('ram_gb', 'N/A')} GB[/cyan]")
        console.print(f"  Python: [cyan]{hw.get('python_version', 'N/A')}[/cyan]")
        console.print(f"  GPUs: [cyan]{hw.get('gpu_count', 0)}[/cyan]")
        if hw.get("gpu_name"):
            console.print(f"  GPU Name: [cyan]{hw.get('gpu_name')}[/cyan]")
            console.print(f"  GPU Memory: [cyan]{hw.get('gpu_memory_gb', 0)} GB[/cyan]")
        console.print(
            f"  CUDA: [cyan]{'Available' if hw.get('cuda_available') else 'Not available'}[/cyan]"
        )
        if hw.get("cuda_version"):
            console.print(f"  CUDA Version: [cyan]{hw.get('cuda_version')}[/cyan]")
    console.print()

    # 3. Credentials File Check
    console.print("[bold]3. Credentials Configuration[/bold]")
    creds_ok, creds_msg = check_credentials_file()
    console.print(f"  {format_check(creds_ok, creds_msg)}")
    if not creds_ok:
        console.print(get_fix_suggestion("credentials", creds_ok))
        all_passed = False
    console.print()

    # 4. Python Dependencies Check
    console.print("[bold]4. Python Dependencies[/bold]")
    deps = check_python_dependencies()
    deps_all_ok = True
    for pkg, (ok, msg) in deps.items():
        console.print(f"  {format_check(ok, msg)}")
        if not ok:
            console.print(get_fix_suggestion(pkg, ok))
            deps_all_ok = False
    if not deps_all_ok:
        all_passed = False
    console.print()

    # 5. nvidia-smi Check
    console.print("[bold]5. NVIDIA Driver (nvidia-smi)[/bold]")
    nvidia_ok, nvidia_msg = check_nvidia_smi()
    console.print(f"  {format_check(nvidia_ok, nvidia_msg)}")
    if not nvidia_ok:
        console.print(get_fix_suggestion("nvidia-smi", nvidia_ok))
        all_passed = False
    console.print()

    # Summary
    if all_passed:
        console.print(Panel.fit("[green]✓ All checks passed[/green]", border_style="green"))
    else:
        console.print(Panel.fit("[yellow]⚠ Some checks failed - see hints above for fixes[/yellow]", border_style="yellow"))


if __name__ == "__main__":
    run_doctor()
