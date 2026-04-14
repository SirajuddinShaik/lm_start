#!/usr/bin/env python3
"""
PM2 Deployment Manager for vLLM Models

Usage:
    ./deploy.py start [speed|balanced|quality]
    ./deploy.py stop
    ./deploy.py restart [speed|balanced|quality]
    ./deploy.py status
    ./deploy.py logs [speed|balanced|quality]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def get_hf_home() -> str:
    return os.environ.get("HF_HOME", "/data/.cache/huggingface")


class PM2Deployer:
    """Manages PM2 deployments with profile selection."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model_id = self._get_model_id()
        self.app_name = self._get_app_name()

    def _get_model_id(self) -> str:
        """Get model ID from model_info.json."""
        info_file = self.model_dir / ".llm-context" / "model-context" / "model_info.json"
        if info_file.exists():
            with open(info_file) as f:
                return json.load(f).get("model_id", "unknown")
        return "unknown"

    def _get_app_name(self) -> str:
        return self.model_id.replace("/", "_").replace("-", "_").lower()

    def _get_visible_devices(self) -> str:
        device_config_file = self.model_dir / ".llm-context" / "model-context" / "device_config.json"
        if device_config_file.exists():
            try:
                with open(device_config_file) as f:
                    data = json.load(f)
                return data.get("gpus", {}).get("visible_devices", "0,1,2,3,4,5,6,7")
            except Exception:
                pass
        return "0,1,2,3,4,5,6,7"

    def _get_profile_config(self, profile: str) -> Dict:
        """Get configuration for a specific profile."""
        # Try to load from vllm_config_best.json first
        best_config_file = self.model_dir / "vllm_config_best.json"
        if best_config_file.exists():
            with open(best_config_file) as f:
                data = json.load(f)
                if data.get("config", {}).get("_profile") == profile or (
                    profile == "speed"
                    and data.get("config", {}).get("_profile") == "performance"
                ):
                    return data.get("config", {})

        # Load all run results and find matching profile
        runs_dir = self.model_dir / ".runs"
        if runs_dir.exists():
            for run_dir in runs_dir.glob("run_*"):
                result_file = run_dir / "result.json"
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                        config = data.get("config", {})
                        config_profile = config.get("_profile", "")
                        if config_profile == profile or (
                            profile == "speed" and config_profile == "performance"
                        ):
                            return config

        # Fallback: load from profile-specific files
        profile_file = self.model_dir / f"vllm_config_{profile}.json"
        if profile_file.exists():
            with open(profile_file) as f:
                return json.load(f)

        raise ValueError(f"No configuration found for profile: {profile}")

    def _build_pm2_config(self, profile: str) -> Dict:
        config = self._get_profile_config(profile)

        args = [
            "serve",
            self.model_id,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]

        for key, value in config.items():
            if key.startswith("_"):
                continue

            flag = f"--{key.replace('_', '-')}"

            if isinstance(value, bool):
                if value:
                    args.append(flag)
            elif value is not None:
                args.extend([flag, str(value)])

        tp = config.get("tensor_parallel_size", 8)
        visible_devices = self._get_visible_devices()
        devices = visible_devices.split(",")[:tp]

        return {
            "apps": [
                {
                    "name": f"{self.app_name}_{profile}",
                    "script": str(self.model_dir / ".venv" / "bin" / "vllm"),
                    "args": args,
                    "interpreter": "none",
                    "cwd": str(self.model_dir),
                    "instances": 1,
                    "autorestart": True,
                    "watch": False,
                    "env": {
                        "HF_HOME": get_hf_home(),
                        "CUDA_VISIBLE_DEVICES": ",".join(devices),
                        "VLLM_ATTENTION_BACKEND": "FLASHINFER",
                        "CUDA_HOME": "/usr/local/cuda-12.9",
                        "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64:/usr/lib/x86_64-linux-gnu",
                        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                        "TORCH_CUDA_MATMUL_PRECISION": "high",
                        "PYTHONHASHSEED": "0",
                        "PROMETHEUS_MULTIPROC_DIR": "/tmp/vllm_prometheus_multiproc",
                    },
                    "log_date_format": "YYYY-MM-DD HH:mm:ss Z",
                    "error_file": str(
                        self.model_dir / "logs" / f"{self.app_name}_{profile}.err.log"
                    ),
                    "out_file": str(
                        self.model_dir / "logs" / f"{self.app_name}_{profile}.out.log"
                    ),
                    "merge_logs": False,
                    "exp_backoff_restart_delay": 100,
                    "max_restarts": 10,
                    "min_uptime": "10s",
                    "restart_delay": 1000,
                    "kill_timeout": 60000,
                }
            ]
        }

    def _save_ecosystem_config(self, profile: str) -> Path:
        """Save ecosystem config for a profile."""
        config = self._build_pm2_config(profile)
        config_file = self.model_dir / f"ecosystem.{profile}.config.js"

        with open(config_file, "w") as f:
            f.write("module.exports = ")
            json.dump(config, f, indent=2)
            f.write(";\n")

        return config_file

    def _check_pm2(self):
        """Check if PM2 is installed."""
        try:
            subprocess.run(["pm2", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: pm2 not found. Install with: npm install -g pm2")
            sys.exit(1)

    def _wait_for_health(self, port: int = 8000, timeout: int = 300) -> bool:
        """Wait for vLLM health endpoint to respond."""
        import time
        import urllib.request
        import urllib.error

        url = f"http://localhost:{port}/health"
        start = time.time()

        print(f"Waiting for health check at {url} (timeout: {timeout}s)...")

        while time.time() - start < timeout:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        elapsed = time.time() - start
                        print(f"✓ Server healthy after {elapsed:.1f}s")
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError, Exception):
                pass
            time.sleep(2)

        print(f"✗ Health check timeout after {timeout}s")
        return False

    def start(self, profile: str = "balanced"):
        """Start server with a specific profile."""
        self._check_pm2()

        print(f"Starting {self.model_id} with {profile} profile...")

        # Save config
        config_file = self._save_ecosystem_config(profile)
        print(f"Config saved to: {config_file}")

        # Create logs directory
        (self.model_dir / "logs").mkdir(exist_ok=True)

        # Check if already running
        app_name = f"{self.app_name}_{profile}"
        result = subprocess.run(
            ["pm2", "describe", app_name], capture_output=True, text=True
        )

        if result.returncode == 0:
            print(f"Server already exists in PM2. Use 'restart {profile}' instead.")
            return

        # Start with PM2
        subprocess.run(["pm2", "start", str(config_file)], check=True)

        # Wait for health check
        port = self._get_port_from_config(profile)
        if self._wait_for_health(port):
            print(f"\n✓ Server started with {profile} profile")
            print(f"  App name: {app_name}")
            print(f"  URL: http://localhost:{port}")
            print(f"\nCheck status: ./deploy.py status")
            print(f"View logs: ./deploy.py logs {profile}")
        else:
            print(f"\n⚠ Server started but health check failed")
            print(f"  Check logs: ./deploy.py logs {profile}")

    def _get_port_from_config(self, profile: str) -> int:
        """Get port from config or return default."""
        try:
            config = self._get_profile_config(profile)
            return config.get("port", 8000)
        except:
            return 8000

    def stop(self, profile: Optional[str] = None):
        """Stop server(s)."""
        self._check_pm2()

        if profile:
            app_name = f"{self.app_name}_{profile}"
            print(f"Stopping {app_name}...")
            subprocess.run(["pm2", "stop", app_name], capture_output=True)
        else:
            # Stop all profiles
            print(f"Stopping all {self.app_name} instances...")
            subprocess.run(["pm2", "stop", f"{self.app_name}_*"], capture_output=True)

        print("✓ Server stopped")

    def restart(self, profile: str = "balanced"):
        """Restart server with a specific profile."""
        self._check_pm2()

        app_name = f"{self.app_name}_{profile}"
        config_file = self.model_dir / f"ecosystem.{profile}.config.js"

        # Check if exists
        result = subprocess.run(
            ["pm2", "describe", app_name], capture_output=True, text=True
        )

        if result.returncode == 0:
            # Restart existing
            print(f"Restarting {app_name}...")
            subprocess.run(["pm2", "restart", app_name], check=True)
        else:
            # Start new
            print(f"Server not running. Starting with {profile} profile...")
            self.start(profile)
            return

        print(f"✓ Server restarted with {profile} profile")

    def status(self):
        """Show status of all servers."""
        self._check_pm2()

        print(f"\nStatus for {self.model_id}:")
        subprocess.run(["pm2", "describe", f"{self.app_name}_*"])

        # Show GPU status
        try:
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            )
        except:
            pass

    def logs(self, profile: Optional[str] = None, lines: int = 100):
        """View logs."""
        self._check_pm2()

        if profile:
            app_name = f"{self.app_name}_{profile}"
            subprocess.run(["pm2", "logs", app_name, "--lines", str(lines)])
        else:
            subprocess.run(["pm2", "logs", f"{self.app_name}_*", "--lines", str(lines)])

    def list_profiles(self):
        """List available profiles."""
        print(f"\nAvailable profiles for {self.model_id}:")
        print("")

        # Check for saved configs
        for profile in ["speed", "balanced", "quality"]:
            config_file = self.model_dir / f"vllm_config_{profile}.json"
            best_file = self.model_dir / "vllm_config_best.json"

            status = ""
            if config_file.exists():
                status = "✓ Saved"

            # Check if it's the best config
            if best_file.exists():
                with open(best_file) as f:
                    data = json.load(f)
                    config_profile = data.get("config", {}).get("_profile", "")
                    if config_profile == profile or (
                        profile == "speed" and config_profile == "performance"
                    ):
                        status += " ★ BEST"

            print(f"  {profile:12} {status}")

        print("")
        print("Usage:")
        print("  ./deploy.py start            # Start with balanced profile (default)")
        print("  ./deploy.py start speed      # Start with speed profile")
        print("  ./deploy.py start balanced   # Start with balanced profile")
        print("  ./deploy.py start quality    # Start with quality profile")


def main():
    parser = argparse.ArgumentParser(
        description="PM2 Deployment Manager for vLLM Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./deploy.py start              # Start with balanced profile (default)
  ./deploy.py start speed        # Start with speed profile
  ./deploy.py start balanced     # Start with balanced profile
  ./deploy.py start quality      # Start with quality profile
  ./deploy.py stop               # Stop all profiles
  ./deploy.py restart            # Restart with balanced profile
  ./deploy.py restart quality    # Restart with quality profile
  ./deploy.py status             # Show status
  ./deploy.py logs               # View logs
  ./deploy.py list               # List available profiles
        """,
    )

    parser.add_argument(
        "command",
        choices=["start", "stop", "restart", "status", "logs", "list"],
        help="Command to execute",
    )

    parser.add_argument(
        "profile",
        nargs="?",
        choices=["speed", "balanced", "quality"],
        default="balanced",
        help="Configuration profile (default: balanced)",
    )

    parser.add_argument(
        "--lines",
        "-n",
        type=int,
        default=100,
        help="Number of log lines to show (default: 100)",
    )

    args = parser.parse_args()

    # Get model directory (current directory or from env)
    model_dir = Path.cwd()

    deployer = PM2Deployer(str(model_dir))

    if args.command == "start":
        deployer.start(args.profile)
    elif args.command == "stop":
        deployer.stop(args.profile)
    elif args.command == "restart":
        deployer.restart(args.profile)
    elif args.command == "status":
        deployer.status()
    elif args.command == "logs":
        deployer.logs(args.profile, args.lines)
    elif args.command == "list":
        deployer.list_profiles()


if __name__ == "__main__":
    main()
