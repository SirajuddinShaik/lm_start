"""Hardware detection module for system information."""

import subprocess
import sys
from typing import Dict, Any


def detect_hardware() -> Dict[str, Any]:
    """
    Detect hardware specifications of the system.

    Returns a dictionary containing:
        - gpu_count: int (0 if no GPU/nvidia-smi not available)
        - gpu_memory_gb: float (total memory in GB, 0 if no GPU)
        - gpu_name: str (name of GPU, empty if no GPU)
        - cpu_cores: int (physical cores)
        - ram_gb: float (total RAM in GB)
        - python_version: str
        - cuda_available: bool
        - cuda_version: str (empty if not available)
    """
    result = {
        "gpu_count": 0,
        "gpu_memory_gb": 0.0,
        "gpu_name": "",
        "cpu_cores": 0,
        "ram_gb": 0.0,
        "python_version": "",
        "cuda_available": False,
        "cuda_version": "",
    }

    # Python version
    result["python_version"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # CPU and RAM using psutil
    try:
        import psutil

        result["cpu_cores"] = (
            psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 0
        )
        result["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        # Fallback if psutil not available
        try:
            result["cpu_cores"] = (
                subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip()
                or 0
            )
            if result["cpu_cores"]:
                result["cpu_cores"] = int(result["cpu_cores"])
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Try to get RAM from /proc/meminfo on Linux
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_kb = int(line.split()[1])
                        result["ram_gb"] = round(mem_kb / (1024**2), 2)
                        break
        except (FileNotFoundError, PermissionError, ValueError):
            pass

    # GPU detection via nvidia-smi
    gpu_info = detect_gpu()
    result["gpu_count"] = gpu_info["count"]
    result["gpu_memory_gb"] = gpu_info["memory_gb"]
    result["gpu_name"] = gpu_info["name"]

    # CUDA detection
    cuda_info = detect_cuda()
    result["cuda_available"] = cuda_info["available"]
    result["cuda_version"] = cuda_info["version"]

    return result


def detect_gpu() -> Dict[str, Any]:
    """
    Detect GPU information using nvidia-smi.

    Returns:
        - count: int (number of GPUs)
        - memory_gb: float (total memory in GB)
        - name: str (name of first GPU, empty if no GPU)
    """
    result = {
        "count": 0,
        "memory_gb": 0.0,
        "name": "",
    }

    try:
        # Try to get GPU info using nvidia-smi
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if output.returncode != 0:
            return result

        lines = output.stdout.strip().split("\n")
        if not lines or not lines[0].strip():
            return result

        result["count"] = len(lines)

        # Get info from first GPU
        first_gpu = lines[0].strip()
        if first_gpu:
            parts = first_gpu.split(",")
            if len(parts) >= 1:
                result["name"] = parts[0].strip()
            if len(parts) >= 2:
                result["memory_gb"] = round(float(parts[1].strip()) / 1024, 2)

        # Calculate total memory across all GPUs
        total_memory = 0.0
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        total_memory += float(parts[1].strip()) / 1024
                    except (ValueError, IndexError):
                        pass

        result["memory_gb"] = round(total_memory, 2)

    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        # nvidia-smi not available or failed - gracefully return 0 GPUs
        pass

    return result


def detect_cuda() -> Dict[str, Any]:
    """
    Detect CUDA availability and version.

    Returns:
        - available: bool
        - version: str (empty if not available)
    """
    result = {
        "available": False,
        "version": "",
    }

    # Try to import torch and check CUDA
    try:
        import torch

        if torch.cuda.is_available():
            result["available"] = True
            result["version"] = torch.version.cuda or ""
            return result
    except ImportError:
        pass

    # Try nvcc first (toolkit version is what we need for installation)
    # nvidia-smi shows DRIVER version which can be higher than toolkit
    try:
        output = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10
        )
        if output.returncode == 0:
            for line in output.stdout.split("\n"):
                if "release" in line.lower():
                    # Extract version like "12.1"
                    parts = line.split("release")
                    if len(parts) >= 2:
                        version = parts[1].strip().split(",")[0]
                        result["available"] = True
                        result["version"] = version
                        return result
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback to nvidia-smi (driver version, may not match toolkit)
    try:
        output = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if output.returncode == 0:
            for line in output.stdout.split("\n"):
                if "CUDA Version:" in line:
                    parts = line.split("CUDA Version:")
                    if len(parts) >= 2:
                        version = parts[1].strip().split()[0]
                        result["available"] = True
                        result["version"] = version
                        return result
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return result


if __name__ == "__main__":
    import json

    print(json.dumps(detect_hardware(), indent=2))
