"""
Session management commands for lm-start.

View and interact with opencode agent sessions from optimization runs.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from lm_start import constants

LM_START_AGENT_TYPES = {"planner", "summarizer", "recovery"}
LM_START_PHASES = {
    "init",
    "fetch_info",
    "download",
    "venv",
    "smoke_test",
    "extract_vllm_config",
    "generate_config",
    "optimize",
    "finalize",
}

app = typer.Typer(help="Manage lm-start agent sessions")
console = Console()


@app.command("show")
def show_session(
    session_id: str = typer.Argument(..., help="Session ID or partial ID"),
):
    """Show session details."""
    matches = find_sessions_by_partial_id(session_id)

    if len(matches) == 0:
        console.print(f"[red]No session found matching: {session_id}[/red]")
        raise typer.Exit(code=1)

    if len(matches) > 1:
        console.print(f"[yellow]Multiple sessions match '{session_id}':[/yellow]\n")
        for m in matches[:10]:
            sid = m.get("session_id", "-")
            agent = m.get("agent_type", "-")
            model = m.get("model_name", "-")
            phase = m.get("phase", "-")
            console.print(f"  {sid}  [{agent}] {model} ({phase})")
        if len(matches) > 10:
            console.print(f"  ... and {len(matches) - 10} more")
        console.print(f"\n[dim]Use a longer ID to uniquely identify the session[/dim]")
        raise typer.Exit(code=1)

    session = matches[0]
    full_id = session.get("session_id", "-")

    console.print(
        Panel.fit(
            f"[bold cyan]Session[/bold cyan]",
            border_style="cyan",
        )
    )

    info_table = Table(show_header=False)
    info_table.add_column("Field", style="green")
    info_table.add_column("Value", style="white")

    info_table.add_row("Session ID", full_id)
    info_table.add_row("Agent Type", session.get("agent_type", "-"))
    info_table.add_row("Title", session.get("title", "-"))
    info_table.add_row("Model", session.get("model_name", "-"))
    info_table.add_row("Phase", session.get("phase", "-"))
    info_table.add_row("Run ID", str(session.get("run_id", "-")))

    details = session.get("details", {})
    if details:
        info_table.add_row(
            "Details", ", ".join(f"{k}={v}" for k, v in details.items() if v)
        )

    timestamp = session.get("timestamp", "-")
    if timestamp and len(timestamp) > 8:
        try:
            dt = datetime.fromisoformat(timestamp)
            timestamp = dt.strftime("%Y-%m-%d %H:%M")
        except:
            pass
    info_table.add_row("Time", str(timestamp))

    console.print(info_table)


@app.command("open")
def open_session(
    session_id: str = typer.Argument(..., help="Session ID or partial ID"),
):
    """Open session in opencode TUI."""
    matches = find_sessions_by_partial_id(session_id)

    if len(matches) == 0:
        console.print(f"[red]No session found matching: {session_id}[/red]")
        raise typer.Exit(code=1)

    if len(matches) > 1:
        console.print(f"[yellow]Multiple sessions match '{session_id}':[/yellow]\n")
        for m in matches[:10]:
            sid = m.get("session_id", "-")
            agent = m.get("agent_type", "-")
            model = m.get("model_name", "-")
            phase = m.get("phase", "-")
            console.print(f"  {sid}  [{agent}] {model} ({phase})")
        if len(matches) > 10:
            console.print(f"  ... and {len(matches) - 10} more")
        console.print(f"\n[dim]Use a longer ID to uniquely identify the session[/dim]")
        raise typer.Exit(code=1)

    session = matches[0]
    opencode_bin = "/home/ubuntu/.opencode/bin/opencode"
    full_session_id = session.get("session_id", session_id)
    model_dir = session.get("model_dir", "")

    console.print(f"[cyan]Opening session in opencode TUI...[/cyan]")
    console.print(
        f"[dim]Model: {session.get('model_name', '-')}, Type: {session.get('agent_type', '-')}[/dim]"
    )
    console.print(f"[dim]Press Ctrl+C to exit[/dim]\n")

    cmd = [opencode_bin, "-s", full_session_id]

    import os

    if model_dir and Path(model_dir).exists():
        os.chdir(model_dir)

    os.execvp(opencode_bin, cmd)


def get_models_base() -> Path:
    system_file = constants.SYSTEM_FILE
    if system_file.exists():
        import yaml

        with open(system_file) as f:
            config = yaml.safe_load(f)
        return Path(config.get("paths", {}).get("models_base", "/data/models"))
    return Path("/data/models")


def is_lmstart_session(session: dict) -> bool:
    agent_type = session.get("agent_type", "").lower()
    if agent_type in LM_START_AGENT_TYPES:
        return True
    phase = session.get("phase", "-")
    if phase in LM_START_PHASES:
        return True
    return False


def find_all_sessions(lmstart_only: bool = True) -> list:
    sessions = []
    models_base = get_models_base()

    if not models_base.exists():
        return sessions

    for model_dir in models_base.iterdir():
        if not model_dir.is_dir():
            continue

        sessions_file = model_dir / ".sessions" / "sessions.json"
        if sessions_file.exists():
            with open(sessions_file) as f:
                model_sessions = json.load(f)
                for s in model_sessions:
                    s["model_name"] = model_dir.name
                    if lmstart_only and not is_lmstart_session(s):
                        continue
                    sessions.append(s)

    # Deduplicate: keep only the most recent entry for each session_id
    seen_ids = {}
    for s in sessions:
        sid = s.get("session_id", "")
        ts = s.get("timestamp", "")
        if sid not in seen_ids or ts > seen_ids[sid].get("timestamp", ""):
            seen_ids[sid] = s

    sessions = list(seen_ids.values())
    sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return sessions


def find_sessions_by_partial_id(partial_id: str) -> list:
    sessions = find_all_sessions()
    partial_lower = partial_id.lower()

    matches = []
    seen_ids = set()
    for s in sessions:
        session_id = s.get("session_id", "")
        if session_id == partial_id:
            if session_id not in seen_ids:
                seen_ids.add(session_id)
                matches.append(s)
            continue
        if partial_lower in session_id.lower():
            if session_id not in seen_ids:
                seen_ids.add(session_id)
                matches.append(s)

    return matches


def find_session_by_id(session_id: str) -> Optional[dict]:
    matches = find_sessions_by_partial_id(session_id)
    if len(matches) == 1:
        return matches[0]
    return None


@app.command("list")
def list_sessions(
    model: Optional[str] = typer.Argument(None, help="Filter by model name"),
    agent_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by agent type (planner/summarizer/recovery)"
    ),
    phase: Optional[str] = typer.Option(
        None, "--phase", "-p", help="Filter by phase (optimize/smoke_test/venv/etc)"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum sessions to show"),
    all_sessions: bool = typer.Option(
        False, "--all", "-a", help="Show all sessions including non-lmstart"
    ),
):
    """List lm-start agent sessions."""
    sessions = find_all_sessions(lmstart_only=not all_sessions)

    if model:
        sessions = [
            s for s in sessions if model.lower() in s.get("model_name", "").lower()
        ]

    if agent_type:
        sessions = [
            s for s in sessions if agent_type.lower() in s.get("agent_type", "").lower()
        ]

    if phase:
        sessions = [s for s in sessions if phase.lower() in s.get("phase", "").lower()]

    sessions = sessions[:limit]

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    table = Table(title=f"lm-start Sessions ({len(sessions)} shown)")
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="green", width=10)
    table.add_column("Model", style="blue", width=22)
    table.add_column("Phase", style="magenta", width=10)
    table.add_column("Details", style="yellow", width=25)
    table.add_column("Time", style="dim", width=12)

    for s in sessions:
        session_id = s.get("session_id", "-")

        agent = s.get("agent_type", "-")
        model_name = s.get("model_name", "-")
        phase = s.get("phase", "-")

        run_info = "-"
        if s.get("run_id"):
            run_info = f"run_{s['run_id']}"
        elif s.get("title"):
            run_info = s["title"][:25]
        d = s.get("details", {})
        if d:
            parts = []
            if d.get("success") is not None:
                parts.append("OK" if d["success"] else "FAIL")
            if d.get("experiments_count"):
                parts.append(f"{d['experiments_count']} exp")
            if parts and run_info == "-":
                run_info = ", ".join(parts)

        timestamp = s.get("timestamp", "-")
        if timestamp and len(timestamp) > 8:
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%m/%d %H:%M")
            except:
                pass

        table.add_row(
            session_id,
            agent,
            model_name[:22],
            phase,
            run_info[:25],
            str(timestamp)[:12],
        )

    console.print(table)
    console.print(
        f"\n[dim]Use 'lm-start session show <ID>' to show or 'lm-start session open <ID>' to open in TUI[/dim]"
    )


@app.command("last")
def last_session(
    model: Optional[str] = typer.Argument(None, help="Model name to filter by"),
    open_it: bool = typer.Option(
        False, "--open", "-o", help="Open the last session in TUI"
    ),
):
    """Show or open the most recent session."""
    sessions = find_all_sessions()

    if model:
        sessions = [
            s for s in sessions if model.lower() in s.get("model_name", "").lower()
        ]

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    last = sessions[0]
    session_id = last.get("session_id", "")

    if open_it:
        open_session(session_id)
    else:
        show_session(session_id)


@app.command("delete")
def delete_session(
    session_id: str = typer.Argument(..., help="Session ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a session from the local registry."""
    session = find_session_by_id(session_id)

    if not session:
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise typer.Exit(code=1)

    model_dir = session.get("model_dir", "")
    model_name = session.get("model_name", "-")

    if not force:
        console.print(f"[yellow]About to delete session:[/yellow]")
        console.print(f"  ID: {session_id}")
        console.print(f"  Model: {model_name}")
        console.print(f"  Type: {session.get('agent_type', '-')}")
        confirm = typer.confirm("\nProceed?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    sessions_file = (
        Path(model_dir) / ".sessions" / "sessions.json" if model_dir else None
    )
    if sessions_file and sessions_file.exists():
        with open(sessions_file) as f:
            sessions = json.load(f)
        sessions = [s for s in sessions if s.get("session_id") != session_id]
        with open(sessions_file, "w") as f:
            json.dump(sessions, f, indent=2)
        console.print(f"[green]✓[/green] Deleted session {session_id}")
    else:
        console.print(f"[red]Could not find sessions file[/red]")
        raise typer.Exit(code=1)


@app.command("clear")
def clear_sessions(
    model: Optional[str] = typer.Argument(
        None, help="Model name to clear sessions for"
    ),
    agent_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by agent type"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Clear sessions from the local registry."""
    models_base = get_models_base()

    if model:
        model_dir = models_base / model.replace("/", "-")
        if not model_dir.exists():
            console.print(f"[red]Model directory not found: {model_dir}[/red]")
            raise typer.Exit(code=1)
        sessions_file = model_dir / ".sessions" / "sessions.json"
        if not sessions_file.exists():
            console.print(f"[yellow]No sessions file for model: {model}[/yellow]")
            return
        with open(sessions_file) as f:
            sessions = json.load(f)
        if agent_type:
            to_delete = [
                s
                for s in sessions
                if agent_type.lower() in s.get("agent_type", "").lower()
            ]
        else:
            to_delete = sessions
        if not to_delete:
            console.print("[yellow]No matching sessions to clear.[/yellow]")
            return
        if not force:
            console.print(
                f"[yellow]About to delete {len(to_delete)} session(s) for {model}[/yellow]"
            )
            if agent_type:
                console.print(f"  Filter: type={agent_type}")
            confirm = typer.confirm("\nProceed?")
            if not confirm:
                console.print("[dim]Cancelled.[/dim]")
                return
        if agent_type:
            sessions = [
                s
                for s in sessions
                if agent_type.lower() not in s.get("agent_type", "").lower()
            ]
        else:
            sessions = []
        with open(sessions_file, "w") as f:
            json.dump(sessions, f, indent=2)
        console.print(f"[green]✓[/green] Cleared {len(to_delete)} session(s)")
    else:
        count = 0
        for model_dir in models_base.iterdir():
            if not model_dir.is_dir():
                continue
            sessions_file = model_dir / ".sessions" / "sessions.json"
            if sessions_file.exists():
                with open(sessions_file) as f:
                    sessions = json.load(f)
                if sessions:
                    count += len(sessions)
        if count == 0:
            console.print("[yellow]No sessions to clear.[/yellow]")
            return
        if not force:
            console.print(
                f"[yellow]About to delete ALL {count} session(s) across all models[/yellow]"
            )
            confirm = typer.confirm("\nProceed?")
            if not confirm:
                console.print("[dim]Cancelled.[/dim]")
                return
        cleared = 0
        for model_dir in models_base.iterdir():
            if not model_dir.is_dir():
                continue
            sessions_file = model_dir / ".sessions" / "sessions.json"
            if sessions_file.exists():
                with open(sessions_file) as f:
                    sessions = json.load(f)
                if sessions:
                    cleared += len(sessions)
                    with open(sessions_file, "w") as f:
                        json.dump([], f, indent=2)
        console.print(f"[green]✓[/green] Cleared {cleared} session(s)")


@app.command("sync")
def sync_sessions(
    model: Optional[str] = typer.Argument(None, help="Model to sync sessions for"),
):
    """Sync sessions from opencode CLI to local registry."""
    opencode_bin = "/home/ubuntu/.opencode/bin/opencode"

    try:
        result = subprocess.run(
            [opencode_bin, "session", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        console.print(f"[red]OpenCode not found at {opencode_bin}[/red]")
        raise typer.Exit(code=1)
    except subprocess.TimeoutExpired:
        console.print("[red]OpenCode session list timed out[/red]")
        raise typer.Exit(code=1)

    lines = result.stdout.strip().split("\n")
    imported = 0

    models_base = get_models_base()
    known_models = {}
    for model_dir in models_base.iterdir():
        if (
            model_dir.is_dir()
            and (
                model_dir / ".llm-context" / "model-context" / "model_info.json"
            ).exists()
        ):
            try:
                with open(
                    model_dir / ".llm-context" / "model-context" / "model_info.json"
                ) as f:
                    info = json.load(f)
                model_id = info.get("model_id", "")
                known_models[model_dir.name] = model_id
            except:
                known_models[model_dir.name] = model_dir.name

    def detect_model_from_title(title: str) -> str:
        title_lower = title.lower()
        for model_dir_name, model_id in known_models.items():
            model_short = (
                model_id.split("/")[-1].lower() if "/" in model_id else model_id.lower()
            )
            if model_short in title_lower or model_dir_name.lower().replace(
                "-", ""
            ) in title_lower.replace("-", "").replace(" ", ""):
                return model_dir_name
            model_parts = model_short.replace("-", " ").split()
            if any(p in title_lower for p in model_parts if len(p) > 3):
                return model_dir_name
        for model_dir_name in known_models:
            name_parts = model_dir_name.lower().replace("-", " ").split()
            if len(name_parts) >= 2:
                combined = "".join(name_parts[:2])
                if combined in title_lower.replace(" ", "").replace("-", ""):
                    return model_dir_name
        return "unknown"

    def detect_phase_from_title(title: str) -> str:
        title_lower = title.lower()
        if (
            "optimization" in title_lower
            or "planner" in title_lower
            or "optimize" in title_lower
        ):
            return "optimize"
        if "smoke" in title_lower:
            return "smoke_test"
        if "venv" in title_lower or "virtual" in title_lower:
            return "venv"
        if "download" in title_lower:
            return "download"
        if "recovery" in title_lower:
            return "recovery"
        if "config" in title_lower:
            return "generate_config"
        return "-"

    def detect_agent_type_from_title(title: str) -> str:
        title_lower = title.lower()
        if "planner" in title_lower:
            return "planner"
        if "summarizer" in title_lower or "summarize" in title_lower:
            return "summarizer"
        if "recovery" in title_lower or "fix" in title_lower or "debug" in title_lower:
            return "recovery"
        return "imported"

    model_sessions = {}
    skipped = 0

    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3 and parts[0].startswith("ses_"):
            session_id = parts[0]
            title = " ".join(parts[1:-1])
            updated = parts[-1]

            model_name = detect_model_from_title(title)
            phase = detect_phase_from_title(title)
            agent_type = detect_agent_type_from_title(title)

            if model_name == "unknown":
                skipped += 1
                continue

            if model_name not in model_sessions:
                model_sessions[model_name] = []

            model_sessions[model_name].append(
                {
                    "session_id": session_id,
                    "agent_type": agent_type,
                    "title": title,
                    "phase": phase,
                    "model_name": model_name,
                    "model_id": known_models.get(model_name, model_name),
                    "model_dir": str(models_base / model_name),
                    "timestamp": updated,
                    "opencode_cmd": f"{opencode_bin} --session {session_id}",
                }
            )
            imported += 1

    for model_name, sessions in model_sessions.items():
        if model_name == "unknown":
            continue
        model_dir = models_base / model_name
        sessions_file = model_dir / ".sessions" / "sessions.json"
        sessions_file.parent.mkdir(parents=True, exist_ok=True)

        existing = []
        if sessions_file.exists():
            with open(sessions_file) as f:
                existing = json.load(f)

        existing_ids = {s.get("session_id") for s in existing}
        for s in sessions:
            if s["session_id"] not in existing_ids:
                existing.append(s)

        with open(sessions_file, "w") as f:
            json.dump(existing, f, indent=2)

    console.print(f"[green]✓[/green] Synced {imported} lm-start session(s)")
    if skipped:
        console.print(f"[dim]  Skipped {skipped} non-lmstart session(s)[/dim]")
