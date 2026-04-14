"""Status command - Show deployment status of models."""

import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lm_start.constants import LM_START_DIR

console = Console()


def get_pm2_status() -> List[Dict[str, Any]]:
    """Get status of all PM2 processes."""
    try:
        result = subprocess.run(
            ["pm2", "jlist"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            import json

            return json.loads(result.stdout)
    except (subprocess.SubprocessError, FileNotFoundError, json.JSONDecodeError):
        pass
    return []


def get_deployed_models() -> List[str]:
    """Get list of deployed models from ~/.lm-start/."""
    deployed = []
    config_dir = LM_START_DIR / "config"

    if config_dir.exists():
        # Look for model config files
        for item in config_dir.iterdir():
            if item.is_dir():
                deployed.append(item.name)

    # Also check PM2 processes
    pm2_processes = get_pm2_status()
    for proc in pm2_processes:
        name = proc.get("name", "")
        if name and name not in deployed:
            # Check if it looks like a model name
            if "/" in name:
                deployed.append(name)

    return sorted(deployed)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed info for a specific model."""
    info = {
        "name": model_name,
        "pm2_status": None,
        "uptime": None,
        "cpu_percent": None,
        "memory_mb": None,
        "restarts": None,
    }

    # Check PM2 status
    pm2_processes = get_pm2_status()
    for proc in pm2_processes:
        name = proc.get("name", "")
        if name == model_name or name.replace("_", "/") == model_name:
            pm2_info = proc.get("pm2_env", {})
            info["pm2_status"] = pm2_info.get("status", "unknown")
            info["uptime"] = pm2_info.get("pm_uptime", None)
            info["cpu_percent"] = proc.get("monit", {}).get("cpu", 0)
            info["memory_mb"] = proc.get("monit", {}).get("memory", 0) / (1024 * 1024)
            info["restarts"] = pm2_info.get("restart_time", 0)

            # Parse uptime
            if info["uptime"]:
                try:
                    import time

                    uptime_ts = int(info["uptime"] / 1000)
                    uptime_seconds = time.time() - uptime_ts
                    hours, remainder = divmod(int(uptime_seconds), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    info["uptime_str"] = f"{hours}h {minutes}m {seconds}s"
                except Exception:
                    info["uptime_str"] = "unknown"

            break

    return info


def format_uptime(uptime_str: Optional[str]) -> str:
    """Format uptime for display."""
    if uptime_str:
        return f"[green]{uptime_str}[/green]"
    return "[dim]N/A[/dim]"


def format_status(status: Optional[str]) -> str:
    """Format PM2 status for display."""
    if status == "online":
        return "[green]online[/green]"
    elif status == "stopped":
        return "[red]stopped[/red]"
    elif status == "errored":
        return "[red]errored[/red]"
    elif status == "launching":
        return "[yellow]launching[/yellow]"
    elif status:
        return f"[yellow]{status}[/yellow]"
    return "[dim]not running[/dim]"


def run_status(model: Optional[str] = None):
    """Show deployment status."""
    if model:
        show_model_status(model)
    else:
        show_all_status()


def show_model_status(model: str):
    """Show status for a specific model."""
    info = get_model_info(model)

    # Check if model exists anywhere
    deployed = get_deployed_models()
    model_exists = model in deployed or model.replace("/", "_") in deployed

    if not info["pm2_status"] and not model_exists:
        console.print(f"[red]Model '{model}' not found[/red]")
        console.print("\n[dim]Available models:[/dim]")
        if deployed:
            for m in deployed:
                console.print(f"  - {m}")
        else:
            console.print("  [dim]No deployed models found[/dim]")
        raise typer.Exit(code=1)

    # Create status panel
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    table.add_row("Model", f"[bold]{info['name']}[/bold]")
    table.add_row("Status", format_status(info["pm2_status"]))

    if info["pm2_status"]:
        if info.get("uptime_str"):
            table.add_row("Uptime", format_uptime(info["uptime_str"]))
        if info["cpu_percent"] is not None:
            table.add_row("CPU", f"{info['cpu_percent']:.1f}%")
        if info["memory_mb"] is not None:
            table.add_row("Memory", f"{info['memory_mb']:.1f} MB")
        if info["restarts"] is not None:
            table.add_row("Restarts", str(info["restarts"]))

    console.print(Panel.fit(table, title=f"Model Status: {model}", border_style="blue"))


def show_all_status():
    """Show status of all deployed models."""
    deployed = get_deployed_models()

    if not deployed:
        console.print(
            Panel.fit(
                "[yellow]No deployed models found[/yellow]\n\n"
                "Run [cyan]lm-start setup <model>[/cyan] to deploy a model.",
                title="Deployment Status",
                border_style="yellow",
            )
        )
        return

    # Get PM2 status
    pm2_processes = get_pm2_status()

    # Build status table
    table = Table(title="Deployed Models")
    table.add_column("Model", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("CPU %", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Uptime", justify="right")

    for model in deployed:
        # Find matching PM2 process
        pm2_info = None
        for proc in pm2_processes:
            if proc.get("name") == model or proc.get("name") == model.replace("/", "_"):
                pm2_info = proc
                break

        if pm2_info:
            env = pm2_info.get("pm2_env", {})
            status = env.get("status", "unknown")
            cpu = pm2_info.get("monit", {}).get("cpu", 0)
            mem = pm2_info.get("monit", {}).get("memory", 0) / (1024 * 1024)
            uptime_ts = env.get("pm_uptime", 0)

            # Format uptime
            if uptime_ts:
                import time

                uptime_seconds = time.time() - (uptime_ts / 1000)
                hours, remainder = divmod(int(uptime_seconds), 3600)
                minutes, _ = divmod(remainder, 60)
                uptime_str = f"{hours}h {minutes}m"
            else:
                uptime_str = "-"

            # Format status color
            if status == "online":
                status_str = "[green]online[/green]"
            elif status == "stopped":
                status_str = "[red]stopped[/red]"
            elif status == "errored":
                status_str = "[red]errored[/yellow]"
            else:
                status_str = f"[yellow]{status}[/yellow]"

            table.add_row(model, status_str, f"{cpu:.1f}%", f"{mem:.1f} MB", uptime_str)
        else:
            table.add_row(
                model,
                "[dim]not running[/dim]",
                "-",
                "-",
                "-",
            )

    console.print(Panel.fit(table, border_style="blue"))


import typer

status_app = typer.Typer()


@status_app.command()
def main(
    model: str = typer.Argument(None, help="Model name to show status for"),
):
    """Show deployment status of models."""
    run_status(model)


if __name__ == "__main__":
    status_app()
