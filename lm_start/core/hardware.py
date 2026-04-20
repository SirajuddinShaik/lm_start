"""
Unified Hardware Detection Module

Provides GPU detection, CUDA verification, and memory calculations.
Consolidates functionality from gpu_detector.py and dgx_spark_detector.py
"""

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class GPUInfo:
    """GPU information."""

    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_percent: float
    temperature_celsius: Optional[float] = None

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024.0

    @property
    def memory_used_gb(self) -> float:
        return self.memory_used_mb / 1024.0

    @property
    def memory_free_gb(self) -> float:
        return self.memory_total_gb - self.memory_used_gb


@dataclass
class HardwareInfo:
    """Complete hardware information."""

    gpus: List[GPUInfo]
    cuda_version: Optional[str]
    driver_version: Optional[str]

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def total_vram_gb(self) -> float:
        return sum(gpu.memory_total_gb for gpu in self.gpus)

    @property
    def available_vram_gb(self) -> float:
        return sum(gpu.memory_free_gb for gpu in self.gpus)

    @property
    def visible_devices(self) -> str:
        """Get CUDA_VISIBLE_DEVICES string."""
        return ",".join(str(gpu.index) for gpu in self.gpus)


class HardwareDetector:
    """
    Detects hardware information including GPUs and CUDA.
    """

    def __init__(self):
        self._hardware_info: Optional[HardwareInfo] = None

    def detect(self) -> HardwareInfo:
        """
        Detect all hardware information.

        Returns:
            HardwareInfo with detected GPUs and CUDA info
        """
        gpus = self._detect_gpus()
        cuda_version = self._detect_cuda_version()
        driver_version = self._detect_driver_version()

        self._hardware_info = HardwareInfo(
            gpus=gpus, cuda_version=cuda_version, driver_version=driver_version
        )

        return self._hardware_info

    def _detect_gpus(self) -> List[GPUInfo]:
        """Detect GPUs using nvidia-smi. Respects CUDA_VISIBLE_DEVICES if set."""
        gpus = []
        
        # Check if CUDA_VISIBLE_DEVICES is set
        visible_devices_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_indices = None
        if visible_devices_str:
            try:
                visible_indices = [int(x.strip()) for x in visible_devices_str.split(",")]
                print(f"[INFO] Using CUDA_VISIBLE_DEVICES: {visible_indices}")
            except ValueError:
                print(f"[WARN] Invalid CUDA_VISIBLE_DEVICES: {visible_devices_str}")

        try:
            # Get GPU details
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            for line in result.stdout.strip().split("\n"):
                parts = line.split(", ")
                if len(parts) >= 5:
                    gpu_index = int(parts[0])
                    # Skip if CUDA_VISIBLE_DEVICES is set and this GPU is not in the list
                    if visible_indices is not None and gpu_index not in visible_indices:
                        continue
                    gpus.append(
                        GPUInfo(
                            index=gpu_index,
                            name=parts[1].strip(),
                            memory_total_mb=int(parts[2]),
                            memory_used_mb=int(parts[3]),
                            utilization_percent=float(parts[4]),
                            temperature_celsius=float(parts[5])
                            if len(parts) > 5
                            else None,
                        )
                    )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # nvidia-smi not available
            pass

        return gpus

    def _detect_cuda_version(self) -> Optional[str]:
        """Detect CUDA version."""
        # Try nvcc
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, check=True
            )
            match = re.search(r"release (\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return None

    def _detect_driver_version(self) -> Optional[str]:
        """Detect NVIDIA driver version."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            return (
                result.stdout.strip().split("\n")[0] if result.stdout.strip() else None
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def can_fit_model(
        self,
        model_params_b: float,
        context_length: int,
        dtype: str = "fp16",
        tensor_parallel: int = 1,
    ) -> bool:
        """
        Check if model can fit in available VRAM.

        Args:
            model_params_b: Model parameters in billions
            context_length: Max context length
            dtype: Data type (fp16, bf16, fp8, int8, int4)
            tensor_parallel: Tensor parallel size

        Returns:
            True if model should fit
        """
        if not self._hardware_info:
            self.detect()

        if not self._hardware_info or not self._hardware_info.gpus:
            return False

        # Calculate VRAM per GPU
        vram_per_gpu = self._hardware_info.total_vram_gb / tensor_parallel

        # Estimate memory requirements
        bytes_per_param = self._get_bytes_per_param(dtype)

        # Base model memory (weights)
        weights_gb = model_params_b * bytes_per_param

        # KV cache memory (rough estimate)
        # 2 * num_layers * hidden_size * context_length * batch_size * bytes_per_param
        # Simplified: ~2 bytes per token per parameter dimension
        kv_cache_gb = (context_length / 1024) * 0.5  # Rough estimate

        # Activations and overhead
        overhead_gb = 2.0

        total_gb = weights_gb + kv_cache_gb + overhead_gb

        # Check if fits with 90% utilization limit
        return total_gb < (vram_per_gpu * 0.9)

    def _get_bytes_per_param(self, dtype: str) -> float:
        """Get bytes per parameter for given dtype."""
        dtype_map = {
            "fp32": 4.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "fp8": 1.0,
            "int8": 1.0,
            "int4": 0.5,
        }
        return dtype_map.get(dtype.lower(), 2.0)

    def recommend_tensor_parallel(self, model_params_b: float) -> int:
        """
        Recommend tensor parallel size based on model size.

        Args:
            model_params_b: Model parameters in billions

        Returns:
            Recommended tensor parallel size
        """
        if not self._hardware_info:
            self.detect()

        available_gpus = self._hardware_info.gpu_count if self._hardware_info else 1

        # Recommendations based on model size
        if model_params_b <= 7:
            return min(1, available_gpus)
        elif model_params_b <= 13:
            return min(1, available_gpus)
        elif model_params_b <= 30:
            return min(2, available_gpus)
        elif model_params_b <= 70:
            return min(4, available_gpus)
        else:
            return min(8, available_gpus)

    def get_summary(self) -> Dict[str, Any]:
        """Get hardware summary as dictionary."""
        if not self._hardware_info:
            self.detect()

        if not self._hardware_info:
            return {"error": "No GPUs detected"}

        return {
            "gpu_count": self._hardware_info.gpu_count,
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "memory_gb": round(gpu.memory_total_gb, 2),
                    "memory_used_gb": round(gpu.memory_used_gb, 2),
                    "utilization": gpu.utilization_percent,
                }
                for gpu in self._hardware_info.gpus
            ],
            "total_vram_gb": round(self._hardware_info.total_vram_gb, 2),
            "available_vram_gb": round(self._hardware_info.available_vram_gb, 2),
            "cuda_version": self._hardware_info.cuda_version,
            "driver_version": self._hardware_info.driver_version,
        }


# Global detector instance
detector = HardwareDetector()


def get_hardware_info() -> HardwareInfo:
    """Get hardware information (cached)."""
    return detector.detect()


def detect_hardware() -> HardwareInfo:
    """Force re-detection of hardware."""
    return detector.detect()
