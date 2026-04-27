"""
Setup command for deploying vLLM models.

This command orchestrates the full model setup process:
1. Validate prerequisites (init done, hardware sufficient)
2. Run phases in sequence with progress display
3. Support resume from last phase
4. Support optimization with different profiles
"""

from pathlib import Path
import subprocess
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from lm_start import constants
from lm_start import config as lm_config
from lm_start import system as sys_detect
from lm_start.phases import (
    phase_init,
    phase_fetch_info,
    phase_download,
    phase_venv,
    phase_smoke_test,
    phase_extract_vllm_config,
    phase_generate_config,
    phase_optimize,
    phase_finalize,
)
from lm_start.state import PhaseRunner, can_resume, get_resume_point

console = Console()

PROFILE_MAP = {
    "speed": "performance",
    "balanced": "balanced",
    "quality": "quality",
}


def check_init_done() -> bool:
    """Check if lm-start has been initialized (system.yaml exists)."""
    system_file = constants.SYSTEM_FILE
    if not system_file.exists():
        return False
    try:
        data = lm_config.load_yaml(str(system_file))
        # Check new structure: device.gpu.count
        gpu_count = data.get("device", {}).get("gpu", {}).get("count")
        return gpu_count is not None and gpu_count > 0
    except Exception:
        return False


def check_hardware_sufficient(model_id: str) -> dict:
    """
    Check if hardware is sufficient for the model.

    Returns dict with:
        - sufficient: bool
        - warnings: list of warning messages
        - hardware: detected hardware info
    """
    hardware = sys_detect.detect_hardware()
    warnings = []

    if hardware.get("gpu_count", 0) == 0:
        warnings.append("No GPU detected. vLLM requires CUDA-capable GPU.")

    if not hardware.get("cuda_available"):
        warnings.append("CUDA not available. vLLM requires CUDA.")

    gpu_memory = hardware.get("gpu_memory_gb", 0)
    if gpu_memory < 16:
        warnings.append(
            f"Low GPU memory ({gpu_memory:.1f} GB). Models may fail to load or run slowly."
        )

    sufficient = hardware.get("gpu_count", 0) > 0 and hardware.get(
        "cuda_available", False
    )

    return {
        "sufficient": sufficient,
        "warnings": warnings,
        "hardware": hardware,
    }


def display_hardware_check(hardware_info: dict):
    """Display hardware detection results."""
    hardware = hardware_info["hardware"]
    warnings = hardware_info["warnings"]

    table = Table(title="[bold]Hardware Detection[/bold]", show_header=False)
    table.add_column("Property", style="cyan", width=25)
    table.add_column("Value", style="green")

    if hardware.get("gpu_count", 0) > 0:
        gpu_info = f"{hardware['gpu_count']}x {hardware.get('gpu_name', 'GPU')}"
        gpu_mem = f"{hardware.get('gpu_memory_gb', 0):.1f} GB"
        table.add_row("GPU", gpu_info)
        table.add_row("GPU Memory", gpu_mem)
    else:
        table.add_row("GPU", "[red]No GPU detected[/red]")

    table.add_row("CPU Cores", str(hardware.get("cpu_cores", 0)))
    table.add_row("RAM", f"{hardware.get('ram_gb', 0):.1f} GB")

    if hardware.get("cuda_available"):
        table.add_row("CUDA", hardware.get("cuda_version", "Available"))
    else:
        table.add_row("CUDA", "[red]Not available[/red]")

    console.print(table)

    if warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  • {warning}")


def get_model_dir(model_id: str) -> Path:
    model_name = model_id.replace("/", "-")
    system_config = lm_config.load_yaml(str(constants.SYSTEM_FILE))
    models_base = (
        system_config.get("paths", {}).get("models_base", "/data/models")
        if system_config
        else "/data/models"
    )
    models_dir = Path(models_base)
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / model_name


def run_setup(
    model_id: str,
    profile: str = "balanced",
    optimize: bool = False,
    resume: bool = False,
    deploy: bool = False,
    dry_run: bool = False,
) -> int:
    """
    Run the full setup process.

    Returns exit code: 0 for success, non-zero for failure.
    """
    profile = PROFILE_MAP.get(profile, profile)

    model_dir = str(get_model_dir(model_id))
    model_name = model_id.split("/")[-1]

    console.print(
        Panel.fit(
            f"[bold cyan]Setting up model:[/bold cyan] {model_id}",
            border_style="cyan",
        )
    )

    # Check init done
    console.print("\n[bold]Checking prerequisites...[/bold]")

    if not check_init_done():
        console.print("[red]✗ lm-start not initialized[/red]")
        console.print("[yellow]Please run: lm-start init[/yellow]")
        return 1
    console.print("[green]✓ Initialization confirmed[/green]")

    hardware_check = check_hardware_sufficient(model_id)
    display_hardware_check(hardware_check)

    if not hardware_check["sufficient"]:
        console.print("\n[red]✗ Hardware insufficient for vLLM[/red]")
        console.print("[yellow]Setup cannot proceed without GPU and CUDA.[/yellow]")
        return 1

    runner = None
    if resume:
        if not can_resume(model_dir):
            console.print("\n[yellow]No resume point found - starting fresh[/yellow]")
        else:
            resume_phase = get_resume_point(model_dir)
            console.print(
                f"\n[cyan]Resuming from phase: {resume_phase.value if resume_phase else 'unknown'}[/cyan]"
            )
            try:
                runner = PhaseRunner(model_dir).load()
            except Exception as e:
                console.print(
                    f"[yellow]Could not load state: {e} - starting fresh[/yellow]"
                )

    if runner is None:
        runner = PhaseRunner(model_dir).init(model_id)
        console.print("[cyan]Initialized state tracking[/cyan]")

    phases = [
        (
            "Initialize",
            lambda: phase_init(model_id, model_dir, model_dir, runner, dry_run),
        ),
        (
            "Fetch Model Info",
            lambda: phase_fetch_info(model_id, model_dir, runner, dry_run),
        ),
        (
            "Download Model",
            lambda: phase_download(model_id, model_dir, runner, dry_run),
        ),
        ("Create venv", lambda: phase_venv(model_dir, runner, dry_run)),
        ("Smoke Test", lambda: phase_smoke_test(model_dir, model_id, runner, dry_run)),
        (
            "Extract vLLM Config",
            lambda: phase_extract_vllm_config(model_dir, runner, dry_run),
        ),
        (
            "Generate Config",
            lambda: phase_generate_config(model_dir, runner=runner, dry_run=dry_run),
        ),
    ]

    if optimize:
        phases.append(
            (
                "Optimize",
                lambda: phase_optimize(
                    model_dir, model_id, runner, dry_run, force=False, agentic=True
                ),
            )
        )

    phases.append(
        (
            "Finalize",
            lambda: phase_finalize(
                model_dir, model_id, model_name, runner=runner, dry_run=dry_run
            ),
        )
    )

    console.print("\n[bold]Running setup phases...[/bold]")

    # Split phases: "fast" phases run inside a progress bar; "heavy" phases
    # produce their own Rich output (long-running vLLM processes) so run after.
    HEAVY_PHASES = {"Smoke Test", "Extract vLLM Config", "Optimize", "Finalize"}
    fast_phases = [(n, f) for n, f in phases if n not in HEAVY_PHASES]
    heavy_phases = [(n, f) for n, f in phases if n in HEAVY_PHASES]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Setup Progress", total=len(fast_phases))

        for phase_name, phase_func in fast_phases:
            progress.update(task, description=f"[cyan]{phase_name}...")

            try:
                result = phase_func()

                if result.success:
                    progress.advance(task)
                    if result.message:
                        if (
                            "already completed" in result.message.lower()
                            or "already completed" in result.message.lower()
                        ):
                            console.print(
                                f"  [green]\u2713[/green] {phase_name}: {result.message}"
                            )
                        else:
                            console.print(f"  [green]\u2713[/green] {result.message}")
                else:
                    console.print(f"\n[red]\u2717 Phase '{phase_name}' failed:[/red]")
                    console.print(f"  {result.message}")
                    return 1

            except Exception as e:
                console.print(f"\n[red]\u2717 Phase '{phase_name}' error:[/red]")
                console.print(f"  {str(e)}")
                return 1

    # Heavy phases run outside the progress bar
    for phase_name, phase_func in heavy_phases:
        console.rule(f"[bold]{phase_name}[/bold]", style="dim")
        try:
            result = phase_func()

            if result.success:
                if result.message:
                    console.print(f"  [green]✓[/green] {phase_name}: {result.message}")
            else:
                console.print(f"\n[red]✗ Phase '{phase_name}' failed:[/red]")
                console.print(f"  {result.message}")
                return 1

        except Exception as e:
            console.print(f"\n[red]✗ Phase '{phase_name}' error:[/red]")
            console.print(f"  {str(e)}")
            return 1

    console.print("\n")
    console.print(
        Panel.fit(
            "[bold green]✓ Setup complete![/bold green]\n\n"
            f"Model: {model_id}\n"
            f"Directory: {model_dir}\n"
            f"Profile: {profile}\n"
            f"Optimized: {'Yes' if optimize else 'No'}\n\n"
            "Run [cyan]./model.sh start[/cyan] to start the server.",
            border_style="green",
        )
    )

    if deploy:
        console.print("\n[bold cyan]Deploying with PM2...[/bold cyan]")
        deploy_py = Path(model_dir) / "deploy.py"
        if deploy_py.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(deploy_py), "start", "balanced"],
                    cwd=model_dir,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    console.print("[green]✓ Deployment successful![/green]")
                    console.print(f"  Model running at: http://localhost:8000")
                else:
                    console.print("[yellow]⚠ Deployment may have issues:[/yellow]")
                    console.print(result.stderr or result.stdout)
            except Exception as e:
                console.print(f"[yellow]⚠ Could not auto-deploy: {e}[/yellow]")
        else:
            console.print("[yellow]⚠ deploy.py not found, trying model.sh...[/yellow]")
            try:
                result = subprocess.run(
                    ["./model.sh", "start"],
                    cwd=model_dir,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    console.print("[green]✓ Deployment successful![/green]")
                else:
                    console.print("[yellow]⚠ Deployment may have issues:[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠ Could not auto-deploy: {e}[/yellow]")

    return 0


def run_setup_command(
    model: str = typer.Argument(
        ..., help="Model name to setup (e.g., Qwen/Qwen2.5-32B)"
    ),
    profile: str = typer.Option(
        "balanced",
        "--profile",
        "-p",
        help="Optimization profile: speed|balanced|quality",
        case_sensitive=False,
    ),
    optimize: bool = typer.Option(
        False,
        "--optimize",
        "-o",
        help="Run optimization to find best configuration",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume from last completed phase",
    ),
    deploy: bool = typer.Option(
        False,
        "--deploy",
        "-d",
        help="Deploy with PM2 after setup completes",
    ),
):
    """Setup a vLLM model for deployment."""
    if profile.lower() not in PROFILE_MAP:
        console.print(f"[red]Invalid profile: {profile}[/red]")
        console.print(f"Valid options: {', '.join(PROFILE_MAP.keys())}")
        raise typer.Exit(code=1)

    exit_code = run_setup(
        model_id=model,
        profile=profile.lower(),
        optimize=optimize,
        resume=resume,
        deploy=deploy,
    )

    raise typer.Exit(code=exit_code)


setup_app = typer.Typer(help="Setup vLLM model for deployment")


@setup_app.command()
def main(
    model: str = typer.Argument(
        ..., help="Model name to setup (e.g., Qwen/Qwen2.5-32B)"
    ),
    profile: str = typer.Option(
        "balanced",
        "--profile",
        "-p",
        help="Optimization profile: speed|balanced|quality",
    ),
    optimize: bool = typer.Option(
        False,
        "--optimize",
        "-o",
        help="Run optimization to find best configuration",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume from last completed phase",
    ),
    deploy: bool = typer.Option(
        False,
        "--deploy",
        "-d",
        help="Deploy with PM2 after setup completes",
    ),
):
    """Setup a vLLM model for deployment."""
    if profile.lower() not in PROFILE_MAP:
        console.print(f"[red]Invalid profile: {profile}[/red]")
        console.print(f"Valid options: {', '.join(PROFILE_MAP.keys())}")
        raise typer.Exit(code=1)

    exit_code = run_setup(
        model_id=model,
        profile=profile.lower(),
        optimize=optimize,
        resume=resume,
        deploy=deploy,
    )

    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    setup_app()
