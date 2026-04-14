import os
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from lm_start.config import load_yaml, save_yaml, ensure_config_dir
from lm_start.constants import CREDENTIALS_FILE

env_app = typer.Typer(help="Manage environment variables and credentials")
console = Console()

CREDENTIALS_VERSION = "1.0"


def load_credentials():
    data = load_yaml(CREDENTIALS_FILE)
    if not data:
        return {"version": CREDENTIALS_VERSION, "environment_variables": {}}
    if "environment_variables" not in data:
        data["environment_variables"] = {}
    return data


def save_credentials(data):
    ensure_config_dir()
    save_yaml(CREDENTIALS_FILE, data)


def mask_value(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 4)


@env_app.command("add")
def env_add(
    key: str = typer.Argument(..., help="Environment variable key"),
    value: Optional[str] = typer.Argument(None, help="Environment variable value"),
    from_env: bool = typer.Option(
        False, "--from-env", help="Read value from current environment"
    ),
):
    if not value and not from_env:
        console.print("[bold red]Error:[/bold red] Provide value or use --from-env")
        raise typer.Exit(1)

    if from_env:
        env_value = os.environ.get(key)
        if not env_value:
            console.print(
                f"[bold red]Error:[/bold red] Environment variable '{key}' not set"
            )
            raise typer.Exit(1)
        value = env_value

    assert value is not None
    credentials = load_credentials()
    credentials["environment_variables"][key] = value
    save_credentials(credentials)

    console.print(f"[bold green]Added:[/bold green] {key} = {mask_value(value)}")


@env_app.command("list")
def env_list():
    credentials = load_credentials()
    env_vars = credentials.get("environment_variables", {})

    if not env_vars:
        console.print("[yellow]No credentials stored[/yellow]")
        return

    table = Table(title="Stored Credentials")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in env_vars.items():
        table.add_row(key, mask_value(value))

    console.print(table)


def export_credentials():
    """Export all stored credentials as environment variables."""
    credentials = load_credentials()
    env_vars = credentials.get("environment_variables", {})
    for key, value in env_vars.items():
        os.environ[key] = value
    return env_vars


def get_credentials_env():
    credentials = load_credentials()
    return credentials.get("environment_variables", {})


@env_app.command("remove")
def env_remove(
    key: str = typer.Argument(..., help="Environment variable key to remove"),
):
    credentials = load_credentials()
    env_vars = credentials.get("environment_variables", {})

    if key not in env_vars:
        console.print(f"[bold red]Error:[/bold red] Key '{key}' not found")
        raise typer.Exit(1)

    del env_vars[key]
    credentials["environment_variables"] = env_vars
    save_credentials(credentials)

    console.print(f"[bold green]Removed:[/bold green] {key}")
