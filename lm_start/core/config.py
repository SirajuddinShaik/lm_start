"""
Unified Configuration Module

Single source of truth for all configuration management.
Loads from config/system.yaml and provides typed access to all settings.
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class GPUConfig:
    """GPU configuration."""

    count: int
    names: List[str]
    memory_gb: int
    visible_devices: str
    compute_capability: str


@dataclass
class CUDAConfig:
    """CUDA configuration."""

    version: str
    home: str
    driver_version: str


@dataclass
class DeviceConfig:
    """Device/hardware configuration."""

    name: str
    description: str
    gpu: GPUConfig
    cuda: CUDAConfig
    system_ram_gb: int
    cpu_count: int


@dataclass
class PathConfig:
    """Path configuration."""

    hf_home: str
    models_base: str
    logs_base: str
    temp: str


@dataclass
class VLLMDefaults:
    """vLLM default settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    trust_remote_code: bool = False
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    enable_auto_tool_choice: bool = True
    tool_call_parser: str = "auto"
    quantization: Optional[str] = None
    max_num_seqs: Optional[int] = None


@dataclass
class PM2Config:
    """PM2 process manager configuration."""

    log_date_format: str
    max_restarts: int
    restart_delay: int
    kill_timeout: int
    listen_timeout: int
    min_uptime: str
    log_max_size: str


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    health_check_interval: int
    startup_timeout: int
    max_experiment_time: int
    health_endpoint: str


@dataclass
class VenvConfig:
    """Venv management configuration for experiments."""

    base_path: str = ".venv"
    per_run: bool = True
    reuse_on_success: bool = True
    cleanup_failed: bool = False


@dataclass
class ExperimentConfig:
    """Experiment settings."""

    max_runs: int
    context_targets: List[int]
    test_port_range: List[int]
    expectation: str = ""
    attention_backends: List[str] = field(default_factory=list)
    venv: VenvConfig = field(default_factory=VenvConfig)


@dataclass
class LLMConfig:
    """LLM configuration for agents."""

    model: str
    timeout: int
    temperature: float
    max_tokens: int


@dataclass
class RecoveryConfig:
    """Error recovery settings."""

    enabled: bool
    auto_retry: bool
    use_llm: bool
    max_llm_attempts: int
    strategies: List[str]


class UnifiedConfig:
    """
    Unified configuration manager.

    Loads configuration from config/system.yaml and provides
    typed access to all settings.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize unified config.

        Args:
            config_path: Path to system.yaml. If None, uses default location.
        """
        if config_path is None:
            # Check user config first, then package config
            user_config = Path.home() / ".lm-start" / "config" / "system.yaml"
            if user_config.exists():
                config_path = user_config
            else:
                # Find config relative to this file
                base_dir = Path(__file__).parent.parent
                config_path = base_dir / "config" / "system.yaml"

        self._config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load configuration from YAML file."""
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self._config_path}")

        with open(self._config_path, "r") as f:
            self._raw_config = yaml.safe_load(f)

    def reload(self) -> None:
        """Reload configuration from disk."""
        self._load()

    @property
    def base_dir(self) -> str:
        """Get base directory for models."""
        return self._raw_config["system"]["base_dir"]

    @property
    def max_retries(self) -> int:
        """Get max retry attempts."""
        return self._raw_config["system"]["max_retries"]

    @property
    def phase_timeout(self) -> int:
        """Get phase timeout in seconds."""
        return self._raw_config["system"]["phase_timeout"]

    @property
    def experiment_timeout(self) -> int:
        """Get experiment timeout in seconds."""
        return self._raw_config["system"]["experiment_timeout"]

    @property
    def device(self) -> DeviceConfig:
        """Get device configuration."""
        device_data = self._raw_config["device"]
        return DeviceConfig(
            name=device_data["name"],
            description=device_data["description"],
            gpu=GPUConfig(**device_data["gpu"]),
            cuda=CUDAConfig(**device_data["cuda"]),
            system_ram_gb=device_data["system_ram_gb"],
            cpu_count=device_data["cpu_count"],
        )

    @property
    def paths(self) -> PathConfig:
        """Get path configuration."""
        paths_data = self._raw_config["paths"]
        return PathConfig(**paths_data)

    @property
    def environment(self) -> Dict[str, str]:
        """Get environment variables."""
        return self._raw_config["environment"].copy()

    def get_env_dict(self, model_venv_path: Optional[str] = None) -> Dict[str, str]:
        """Get environment variables as dict for subprocess calls."""
        env = os.environ.copy()

        # If model venv specified, override VIRTUAL_ENV and PATH
        # to prevent inheriting wrong venv from parent shell
        if model_venv_path:
            env["VIRTUAL_ENV"] = model_venv_path
            # Update PATH to put model venv first
            original_path = env.get("PATH", "")
            venv_bin = str(Path(model_venv_path) / "bin")
            # Remove any existing venv paths from PATH
            path_parts = [
                p
                for p in original_path.split(":")
                if ".venv" not in p and "venv" not in p.lower()
            ]
            env["PATH"] = venv_bin + ":" + ":".join(path_parts)

        # Set cache dirs to temp locations to prevent cross-venv contamination
        import tempfile
        cache_base = tempfile.gettempdir()
        env["NUMBA_CACHE_DIR"] = f"{cache_base}/numba_cache"
        env["TRITON_CACHE_DIR"] = f"{cache_base}/triton_cache"
        env["TORCHINDUCTOR_CACHE_DIR"] = f"{cache_base}/torch_inductor_cache"
        env["FLASHINFER_DISABLE_JIT"] = "1"
        env["CUDA_MODULE_LOADING"] = "LAZY"
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # Disable flashinfer all-reduce (falls back to NCCL)
        env["VLLM_USE_FLASHINFER_ALLREDUCE"] = "0"

        env.update(self.environment)
        return env

    @property
    def vllm_defaults(self) -> VLLMDefaults:
        """Get vLLM default configuration."""
        return VLLMDefaults(**self._raw_config["vllm_defaults"])

    @property
    def pm2(self) -> PM2Config:
        """Get PM2 configuration."""
        return PM2Config(**self._raw_config["pm2"])

    @property
    def monitoring(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return MonitoringConfig(**self._raw_config["monitoring"])

    @property
    def experiment(self) -> ExperimentConfig:
        """Get experiment configuration."""
        exp_data = self._raw_config["experiment"].copy()
        # Handle nested venv config
        if "venv" in exp_data and isinstance(exp_data["venv"], dict):
            exp_data["venv"] = VenvConfig(**exp_data["venv"])
        else:
            exp_data["venv"] = VenvConfig()
        # Handle optional attention_backends with defaults
        if "attention_backends" not in exp_data:
            exp_data["attention_backends"] = ["FLASHINFER", "FLASH_ATTN", "XFORMERS"]
        if "expectation" not in exp_data:
            exp_data["expectation"] = ""
        return ExperimentConfig(**exp_data)

    @property
    def llm(self) -> LLMConfig:
        """Get LLM configuration."""
        return LLMConfig(**self._raw_config["llm"])

    @property
    def recovery(self) -> RecoveryConfig:
        """Get recovery configuration."""
        rec_data = self._raw_config["recovery"]
        return RecoveryConfig(**rec_data)

    @property
    def phases(self) -> Dict[str, Any]:
        """Get phases configuration."""
        return self._raw_config["phases"].copy()

    def get_phase_config(self, phase_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific phase."""
        phases = self._raw_config["phases"]
        if phase_name in phases:
            return phases[phase_name].copy()
        return None

    def get_raw(self, key: str, default: Any = None) -> Any:
        """Get raw config value by key path (e.g., 'system.base_dir')."""
        keys = key.split(".")
        value = self._raw_config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


# Global config instance (lazy-loaded)
_config_instance: Optional[UnifiedConfig] = None


def get_config() -> UnifiedConfig:
    """Get the global unified config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = UnifiedConfig()
    return _config_instance


def reload_config() -> UnifiedConfig:
    """Reload and return the global config."""
    global _config_instance
    _config_instance = UnifiedConfig()
    return _config_instance
