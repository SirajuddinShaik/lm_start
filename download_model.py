#!/usr/bin/env python3
"""
Download model from HuggingFace using huggingface-cli.

Supports:
- Pre-download models before serving
- Tmux session for long-running downloads
- Resume interrupted downloads
- Auth token for gated models

Usage:
    python download_model.py <model_id> --model-dir /models/llama

    # With tmux (for long downloads)
    python download_model.py <model_id> --model-dir /models/llama --tmux

    # Check download status
    python download_model.py <model_id> --model-dir /models/llama --status
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


TMUX_PREFIX = "mdl_"


def get_tmux_session_name(model_id: str) -> str:
    model_name = model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    return f"{TMUX_PREFIX}{model_name}"


def tmux_session_exists(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name], capture_output=True
    )
    return result.returncode == 0


def create_tmux_session(session_name: str, command: str, working_dir: str) -> bool:
    try:
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session_name,
                "-c",
                working_dir,
                command,
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create tmux session: {e}")
        return False


def attach_tmux_session(session_name: str) -> None:
    os.system(f"tmux attach -t {session_name}")


def kill_tmux_session(session_name: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)


def check_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
    return token


def check_model_downloaded(model_id: str, hf_home: str) -> bool:
    local_dir_path = Path(hf_home) / model_id
    if local_dir_path.exists():
        safetensors = list(local_dir_path.glob("*.safetensors"))
        pytorch_bins = list(local_dir_path.glob("pytorch_model*.bin"))
        gguf_files = list(local_dir_path.glob("*.gguf"))
        if safetensors or pytorch_bins or gguf_files:
            return True

    cache_model_id = model_id.replace("/", "--")
    hub_path = Path(hf_home) / "hub" / f"models--{cache_model_id}" / "snapshots"
    if hub_path.exists():
        for snapshot in hub_path.iterdir():
            if snapshot.is_dir():
                safetensors = list(snapshot.glob("*.safetensors"))
                pytorch_bins = list(snapshot.glob("pytorch_model*.bin"))
                gguf_files = list(snapshot.glob("*.gguf"))
                if safetensors or pytorch_bins or gguf_files:
                    return True

    return False


def get_model_cache_path(model_id: str, hf_home: str) -> Optional[str]:
    local_dir_path = Path(hf_home) / model_id
    if local_dir_path.exists():
        safetensors = list(local_dir_path.glob("*.safetensors"))
        pytorch_bins = list(local_dir_path.glob("pytorch_model*.bin"))
        gguf_files = list(local_dir_path.glob("*.gguf"))
        if safetensors or pytorch_bins or gguf_files:
            return str(local_dir_path)

    cache_model_id = model_id.replace("/", "--")
    snapshots_path = Path(hf_home) / "hub" / f"models--{cache_model_id}" / "snapshots"
    if snapshots_path.exists():
        for snapshot in snapshots_path.iterdir():
            if snapshot.is_dir():
                safetensors = list(snapshot.glob("*.safetensors"))
                pytorch_bins = list(snapshot.glob("pytorch_model*.bin"))
                gguf_files = list(snapshot.glob("*.gguf"))
                if safetensors or pytorch_bins or gguf_files:
                    return str(snapshot)
    return None


def get_download_size(model_id: str) -> Optional[float]:
    try:
        import requests

        api_url = f"https://huggingface.co/api/models/{model_id}"
        response = requests.get(api_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            total_bytes = 0
            for sibling in data.get("siblings", []):
                filename = sibling.get("rfilename", "")
                if filename.endswith((".safetensors", ".bin", ".gguf")):
                    total_bytes += sibling.get("size", 0)
            if total_bytes > 0:
                return round(total_bytes / (1024**3), 2)
    except Exception:
        pass
    return None


def download_model(
    model_id: str,
    model_dir: str,
    hf_home: str,
    use_tmux: bool = False,
    token: Optional[str] = None,
    include_patterns: Optional[list] = None,
    exclude_patterns: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Download model using huggingface-cli.

    Returns:
        Dict with status, local_path, and session_name (if using tmux)
    """
    result = {
        "status": "unknown",
        "local_path": None,
        "session_name": None,
        "error": None,
    }

    if check_model_downloaded(model_id, hf_home):
        print(f"Model already downloaded: {model_id}")
        local_path = get_model_cache_path(model_id, hf_home)
        result["status"] = "already_downloaded"
        result["local_path"] = local_path
        return result

    cmd = ["hf", "download", model_id]

    if token:
        cmd.extend(["--token", token])

    if include_patterns:
        for pattern in include_patterns:
            cmd.extend(["--include", pattern])

    if exclude_patterns:
        for pattern in exclude_patterns:
            cmd.extend(["--exclude", pattern])

    session_name = get_tmux_session_name(model_id)

    env = os.environ.copy()
    env["HF_HOME"] = hf_home

    if use_tmux:
        if tmux_session_exists(session_name):
            print(f"Tmux session already exists: {session_name}")
            print(f"Attach with: tmux attach -t {session_name}")
            result["status"] = "in_progress"
            result["session_name"] = session_name
            return result

        env_export = f"export HF_HOME={hf_home} && "
        cmd_str = env_export + " ".join(cmd)
        print(f"Starting download in tmux session: {session_name}")
        print(f"Command: {cmd_str}")

        if create_tmux_session(session_name, cmd_str, model_dir):
            print(f"\nDownload started in background.")
            print(f"  Check status: tmux attach -t {session_name}")
            print(
                f"  Or: python download_model.py {model_id} --model-dir {model_dir} --status"
            )
            result["status"] = "started"
            result["session_name"] = session_name
        else:
            result["status"] = "failed"
            result["error"] = "Failed to create tmux session"
    else:
        print(f"Downloading model: {model_id}")
        print(f"HF_HOME: {hf_home}")
        print(f"Command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            for line in iter(process.stdout.readline, ""):
                print(line, end="")

            process.wait()

            if process.returncode == 0:
                print(f"\nDownload completed successfully!")
                result["status"] = "completed"
                result["local_path"] = get_model_cache_path(model_id, hf_home)
            else:
                result["status"] = "failed"
                result["error"] = f"Download failed with code {process.returncode}"

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)

    return result


def check_download_status(model_id: str, hf_home: str) -> Dict[str, Any]:
    """Check the status of a download."""
    session_name = get_tmux_session_name(model_id)
    result = {
        "model_id": model_id,
        "downloaded": False,
        "tmux_session": None,
        "local_path": None,
    }

    if check_model_downloaded(model_id, hf_home):
        result["downloaded"] = True
        result["local_path"] = get_model_cache_path(model_id, hf_home)

    if tmux_session_exists(session_name):
        result["tmux_session"] = session_name

        try:
            capture = subprocess.run(
                ["tmux", "capture-pane", "-t", session_name, "-p"],
                capture_output=True,
                text=True,
            )
            if capture.returncode == 0:
                result["last_output"] = capture.stdout[-500:]
        except Exception:
            pass

    return result


def update_model_info(model_dir: str, updates: Dict[str, Any]) -> None:
    info_path = Path(model_dir) / ".llm-context" / "model-context" / "model_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        info.update(updates)
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Download model from HuggingFace")
    parser.add_argument(
        "model", nargs="?", help="HuggingFace model ID (e.g., meta-llama/Llama-3-70b)"
    )
    parser.add_argument(
        "--model-dir", required=True, help="Model directory (contains .llm-context/)"
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", "/juspfsdata1/.cache/huggingface"),
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--tmux", action="store_true", help="Run download in tmux session"
    )
    parser.add_argument("--status", action="store_true", help="Check download status")
    parser.add_argument(
        "--attach", action="store_true", help="Attach to tmux download session"
    )
    parser.add_argument(
        "--token", help="HuggingFace API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--include", nargs="+", help="File patterns to include (e.g., '*.safetensors')"
    )
    parser.add_argument(
        "--exclude", nargs="+", help="File patterns to exclude (e.g., '*.gguf')"
    )

    args = parser.parse_args()

    if args.status or args.attach:
        model_info_path = (
            Path(args.model_dir) / ".llm-context" / "model-context" / "model_info.json"
        )
        if model_info_path.exists():
            with open(model_info_path) as f:
                model_info = json.load(f)
            model_id = model_info.get("model_id")
        elif args.model:
            model_id = args.model
        else:
            print(
                "ERROR: Cannot determine model ID. Provide --model or ensure model_info.json exists."
            )
            sys.exit(1)
    else:
        if not args.model:
            print("ERROR: Model ID required for download")
            parser.print_help()
            sys.exit(1)
        model_id = args.model

    session_name = get_tmux_session_name(model_id)

    if args.attach:
        if tmux_session_exists(session_name):
            attach_tmux_session(session_name)
        else:
            print(f"No active tmux session: {session_name}")
        return

    if args.status:
        status = check_download_status(model_id, args.hf_home)
        print(f"\nDownload Status: {model_id}")
        print(f"  Downloaded: {status['downloaded']}")
        if status["local_path"]:
            print(f"  Local path: {status['local_path']}")
        if status["tmux_session"]:
            print(f"  Tmux session: {status['tmux_session']}")
            print(f"  Attach with: tmux attach -t {status['tmux_session']}")
            if status.get("last_output"):
                print(f"\n  Last output:\n  {status['last_output']}")
        return

    token = args.token or check_hf_token()

    result = download_model(
        model_id=model_id,
        model_dir=args.model_dir,
        hf_home=args.hf_home,
        use_tmux=args.tmux,
        token=token,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
    )

    if result["status"] in ("completed", "already_downloaded"):
        update_model_info(
            args.model_dir,
            {"download_status": "completed", "local_model_path": result["local_path"]},
        )

    if result["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
