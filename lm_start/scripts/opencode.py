"""OpenCode installer integration for lm-start."""

import os
import subprocess
from pathlib import Path

from lm_start.constants import LM_START_DIR

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


def install(version: str = "1.14.19") -> bool:
    """Install OpenCode to ~/.lm-start/opencode/.

    Downloads and extracts from anomalyco/opencode GitHub releases.

    Args:
        version: Version to install (default: "1.14.19")

    Returns:
        True if installation succeeded, False otherwise.
    """
    import tarfile
    import tempfile
    import shutil

    OPENCODE_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    OPENCODE_BIN_DIR.mkdir(parents=True, exist_ok=True)

    # Use specific release from anomalyco/opencode
    # GitHub releases URL pattern: /releases/download/{tag}/{asset}
    download_url = f"https://github.com/anomalyco/opencode/releases/download/v{version}/opencode-linux-x64.tar.gz"

    print(f"Downloading OpenCode from GitHub (anomalyco/opencode)...")

    try:
        # Create temp file for download
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        # Try curl first
        result = subprocess.run(
            ["curl", "-sL", "--fail", "-o", tmp_path, download_url],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            # Try wget as fallback
            result = subprocess.run(
                ["wget", "-q", "-O", tmp_path, download_url],
                capture_output=True,
                text=True,
                timeout=120,
            )

        if result.returncode != 0:
            print(f"Failed to download OpenCode")
            print(f"  Exit code: {result.returncode}")
            print(f"  stderr: {result.stderr}")
            print(f"  URL: {download_url}")
            return False

        # Extract tarball
        print("Extracting OpenCode...")
        with tempfile.TemporaryDirectory() as extract_dir:
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(extract_dir)

            # Find the opencode binary in extracted files
            extracted_bin = None
            for root, dirs, files in os.walk(extract_dir):
                if "opencode" in files:
                    extracted_bin = os.path.join(root, "opencode")
                    break

            if not extracted_bin:
                print("Could not find opencode binary in archive")
                return False

            # Copy to destination
            shutil.copy2(extracted_bin, OPENCODE_BINARY)
            os.chmod(OPENCODE_BINARY, 0o755)

        # Cleanup
        os.unlink(tmp_path)

        if is_installed():
            installed_version = get_version()
            print(f"✓ OpenCode v{installed_version} installed successfully")
            return True
        else:
            print(f"Installation failed - binary not found at {OPENCODE_BINARY}")
            return False

    except subprocess.TimeoutExpired:
        print("Installation timed out")
        return False
    except Exception as e:
        print(f"Installation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def ensure_version(version: str = "1.14.19") -> bool:
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
