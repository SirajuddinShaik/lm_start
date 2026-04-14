import typer
from rich.console import Console
from typing import Optional

from lm_start import __version__
from lm_start.commands.env import env_app
from lm_start.commands.status import run_status
from lm_start.commands.logs import run_logs
from lm_start.commands.session import app as session_app

app = typer.Typer()
console = Console()

app.add_typer(env_app, name="env")
app.add_typer(session_app, name="session")


@app.command()
def version():
    """Show version."""
    typer.echo(f"lm-start version {__version__}")


@app.command()
def init():
    """Initialize lm-start configuration."""
    from lm_start.commands.init import run_init

    run_init()


@app.command()
def setup(
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
    """Setup model deployment."""
    from lm_start.commands.setup import run_setup

    exit_code = run_setup(
        model_id=model,
        profile=profile.lower(),
        optimize=optimize,
        resume=resume,
        deploy=deploy,
    )
    raise typer.Exit(code=exit_code)


@app.command()
def doctor():
    """Run system diagnostics."""
    from lm_start.commands.doctor import run_doctor

    run_doctor()


@app.command()
def status(
    model: Optional[str] = typer.Argument(None, help="Model name to show status for"),
):
    """Show deployment status of models."""
    run_status(model)


@app.command()
def logs(
    model: str = typer.Argument(..., help="Model name to show logs for"),
    lines: int = typer.Option(100, "--lines", "-n", help="Number of log lines to show"),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Follow logs in real-time"
    ),
):
    """Show deployment logs for a model."""
    run_logs(model, lines, follow)


if __name__ == "__main__":
    app()
