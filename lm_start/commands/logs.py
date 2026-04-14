"""Logs command - Show deployment logs for models."""

import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from lm_start.commands.status import get_deployed_models

console = Console()


def get_pm2_log_path(model_name: str) -> Optional[Path]:
    """Get the log file path for a PM2 process."""
    try:
        result = subprocess.run(
            ["pm2", "describe", model_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "err log path" in line.lower() or "out log path" in line.lower():
                    parts = line.split(":")
                    if len(parts) >= 2:
                        log_path = Path(parts[-1].strip())
                        if log_path.exists():
                            return log_path
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Fallback: check common PM2 log paths
    home = Path.home()
    common_paths = [
        home / ".pm2" / "logs" / f"{model_name}-out.log",
        home / ".pm2" / "logs" / f"{model_name}-err.log",
        home / ".pm2" / "logs" / f"{model_name.replace('/', '_')}-out.log",
        home / ".pm2" / "logs" / f"{model_name.replace('/', '_')}-err.log",
    ]

    for path in common_paths:
        if path.exists():
            return path

    return None


def read_log_lines(log_path: Path, lines: int) -> str:
    """Read last N lines from a log file."""
    try:
        with open(log_path, "r") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"Error reading log file: {e}"


def follow_logs(model_name: str):
    """Follow logs in real-time."""
    log_path = get_pm2_log_path(model_name)

    if not log_path:
        console.print(f"[red]No logs found for model '{model_name}'[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[bold blue]Following logs for {model_name} (Ctrl+C to exit)[/bold blue]"
    )
    console.print(f"[dim]Log file: {log_path}[/dim]\n")

    try:
        # Get initial line count
        with open(log_path, "r") as f:
            last_pos = f.seek(0, 2)

        with Live(console=console, refresh_per_second=4, transient=True) as live:
            while True:
                import time

                with open(log_path, "r") as f:
                    f.seek(last_pos)
                    new_content = f.read()
                    if new_content:
                        live.update(new_content.strip())
                    last_pos = f.tell()
                time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped following logs[/dim]")


def run_logs(model: str, lines: int = 100, follow: bool = False):
    """Show deployment logs for a model."""
    if follow:
        follow_logs(model)
        return

    deployed = get_deployed_models()
    model_exists = model in deployed or model.replace("_", "/") in deployed

    log_path = get_pm2_log_path(model)

    if not log_path and not model_exists:
        console.print(f"[red]Model '{model}' not found[/red]")
        console.print("\n[dim]Available models:[/dim]")
        if deployed:
            for m in deployed:
                console.print(f"  - {m}")
        else:
            console.print("  [dim]No deployed models found[/dim]")
        raise typer.Exit(code=1)

    if not log_path:
        console.print(f"[yellow]No logs found for model '{model}'[/yellow]")
        console.print(
            "[dim]Model may not be running or PM2 logs may not be available[/dim]"
        )
        raise typer.Exit(code=1)

    log_content = read_log_lines(log_path, lines)

    if log_content:
        console.print(
            Panel.fit(
                log_content,
                title=f"Logs: {model} (last {lines} lines)",
                border_style="blue",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[dim]No log content available[/dim]",
                title=f"Logs: {model}",
                border_style="blue",
            )
        )


logs_app = typer.Typer()


@logs_app.command()
def main(
    model: str = typer.Argument(..., help="Model name to show logs for"),
    lines: int = typer.Option(100, "--lines", "-n", help="Number of log lines to show"),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Follow logs in real-time"
    ),
):
    """Show deployment logs for a model."""
    run_logs(model, lines, follow)


if __name__ == "__main__":
    logs_app()
