"""Utils module."""

from .logger import get_logger
from .model_utils import (
    get_hf_home,
    get_model_cache_path,
    read_model_config,
    get_max_position_embeddings,
    calculate_optimal_config,
    copy_all_model_files,
    detect_model_capabilities,
)
from .theoretical_calculator import (
    TheoreticalCalculator,
    HardwareConfig,
    ModelConfig,
    calculate_from_model_info,
    get_model_size_from_id,
)
from .benchmark_runner import BenchmarkRunner, save_benchmark_results
from .prompt_manager import get_prompt, get_mock_response
from .vllm_flag_selector import VLLMFlagSelector, OptimizationProfile
from .capacity_calculator import (
    CapacityCalculator,
    CapacityEstimate,
    calculate_capacity,
)
from .intelligent_config import IntelligentConfigGenerator, generate_config
from .adaptive_validator import AdaptiveValidator, ValidationResult

__all__ = [
    # Logger
    "get_logger",
    # Model utilities
    "get_model_cache_path",
    "read_model_config",
    "get_max_position_embeddings",
    "calculate_optimal_config",
    "copy_all_model_files",
    "detect_model_capabilities",
    # Theoretical calculator
    "TheoreticalCalculator",
    "HardwareConfig",
    "ModelConfig",
    "calculate_from_model_info",
    "get_model_size_from_id",
    # Benchmark runner
    "BenchmarkRunner",
    "save_benchmark_results",
    # Prompt manager
    "get_prompt",
    "get_mock_response",
    # VLLM flag selector
    "VLLMFlagSelector",
    "OptimizationProfile",
    # Capacity calculator (NEW)
    "CapacityCalculator",
    "CapacityEstimate",
    "calculate_capacity",
    # Intelligent config (NEW)
    "IntelligentConfigGenerator",
    "generate_config",
    # Adaptive validator (NEW)
    "AdaptiveValidator",
    "ValidationResult",
]
