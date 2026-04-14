"""OpenCode installer integration for lm-start."""

import os
import subprocess
from pathlib import Path

from .constants import LM_START_DIR

OPENCODE_INSTALL_DIR = LM_START_DIR / "opencode"
OPENCODE_BIN_DIR = OPENCODE_INSTALL_DIR / "bin"
OPENCODE_BINARY = OPENCODE_BIN_DIR / "opencode"


def is_installed() -> bool:
    """Check if OpenCode is installed at the expected location.

    Returns:
        True if OpenCode binary exists and is executable, False otherwise.
    """
    return OPENCODE_BINARY.exists() and os.access(OPENCODE_BINARY, os.X_OK)


def get_version() -> str | None:
    """Get the installed OpenCode version.

    Returns:
        Version string (e.g., "1.4.0") or None if not installed.
    """
    if not is_installed():
        return None

    try:
        result = subprocess.run(
            [str(OPENCODE_BINARY), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip() or result.stderr.strip()

        if output:
            for part in output.split():
                if part.startswith("v") and "." in part:
                    return part[1:]
                elif "." in part and part.replace(".", "").isdigit():
                    return part

        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def install(version: str = "1.4.0") -> bool:
    """Install OpenCode to ~/.lm-start/opencode/.

    Uses the official install script with specified version and install directory.

    Args:
        version: Version to install (default: "1.4.0")

    Returns:
        True if installation succeeded, False otherwise.
    """
    OPENCODE_INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["VERSION"] = version
    env["OPENCODE_INSTALL_DIR"] = str(OPENCODE_INSTALL_DIR)

    def try_download(tools: list[str]) -> str | None:
        for tool in tools:
            result = None
            try:
                if tool == "curl":
                    result = subprocess.run(
                        ["curl", "-sL", "https://get.opencode.ai"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        env=env,
                    )
                elif tool == "wget":
                    result = subprocess.run(
                        ["wget", "-qO-", "https://get.opencode.ai"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        env=env,
                    )

                if result and result.returncode == 0 and result.stdout.strip():
                    return result.stdout
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                continue

        return None

    script = try_download(["curl", "wget"])
    if not script:
        print("Failed to download install script - check network connectivity")
        return False

    try:
        process = subprocess.run(
            ["sh"],
            input=script,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )

        if process.returncode != 0:
            print(f"Install script failed: {process.stderr}")
            return False

        if is_installed():
            installed_version = get_version()
            if installed_version == version:
                return True
            else:
                print(f"Version mismatch: expected {version}, got {installed_version}")
                return False
        else:
            print(f"Installation completed but binary not found at {OPENCODE_BINARY}")
            return False

    except subprocess.TimeoutExpired:
        print("Installation timed out")
        return False
    except Exception as e:
        print(f"Installation failed: {e}")
        return False


def ensure_version(version: str = "1.4.0") -> bool:
    """Ensure OpenCode is installed with the specified version.

    Checks if the correct version is already installed and skips
    installation if version matches.

    Args:
        version: Version to ensure (default: "1.4.0")

    Returns:
        True if correct version is installed, False otherwise.
    """
    if is_installed():
        installed_version = get_version()
        if installed_version == version:
            return True
        else:
            print(f"OpenCode version {installed_version} != {version}, reinstalling...")
            return install(version)

    return install(version)
