"""
Profile Configuration Generator

Generates 3 profile variants (performance, quality, balanced) from a base working config.
Each profile optimizes for different goals while maintaining the same base context length.

Usage:
    from utils.profile_config_generator import ProfileConfigGenerator
    
    generator = ProfileConfigGenerator(model_dir, gpu_type="H200")
    
    # Generate all 3 profiles from base config
    profiles = generator.generate_profiles(base_config)
    
    # profiles = {
    #     "performance": {...},
    #     "quality": {...},
    #     "balanced": {...}
    # }
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ProfileType(str, Enum):
    PERFORMANCE = "performance"
    QUALITY = "quality"
    BALANCED = "balanced"


@dataclass
class ProfileDefinition:
    """Definition of an optimization profile."""
    name: str
    description: str
    goals: List[str]
    acceptable_trade_offs: List[str]


class ProfileConfigGenerator:
    """
    Generates 3 profile variants from a base working configuration.
    
    Each profile maintains the same core parameters (context, TP, model)
    but adjusts optimization flags for different goals.
    """
    
    # Hardware capabilities
    FP8_CAPABLE_GPUS = ["H100", "H200", "H800", "MI300X"]
    
    def __init__(self, model_dir: Path, gpu_type: Optional[str] = None):
        """
        Initialize generator.
        
        Args:
            model_dir: Path to model directory
            gpu_type: GPU type (e.g., "H200") for hardware-specific optimizations
        """
        self.model_dir = Path(model_dir)
        self.gpu_type = gpu_type or "unknown"
        self.supports_fp8 = any(g in self.gpu_type.upper() for g in self.FP8_CAPABLE_GPUS)
        
        # Load available flags if extracted
        self.available_flags = self._load_available_flags()
    
    def _load_available_flags(self) -> Dict[str, Any]:
        """Load dynamically extracted vLLM flags."""
        flags_path = self.model_dir / "vllm_available_flags.json"
        if flags_path.exists():
            with open(flags_path) as f:
                data = json.load(f)
                return data.get("flags", {})
        return {}
    
    def _flag_exists(self, flag_name: str) -> bool:
        """Check if a flag exists in available flags."""
        return flag_name in self.available_flags
    
    def _get_flag_choices(self, flag_name: str) -> Optional[List[Any]]:
        """Get valid choices for a flag."""
        if flag_name in self.available_flags:
            return self.available_flags[flag_name].get("choices")
        return None
    
    def generate_profiles(self, base_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Generate all 3 profile variants from base config.
        
        Args:
            base_config: Working base configuration (from max-context-first optimization)
        
        Returns:
            Dict with "performance", "quality", "balanced" configs
        """
        return {
            ProfileType.PERFORMANCE: self._generate_performance_profile(base_config),
            ProfileType.QUALITY: self._generate_quality_profile(base_config),
            ProfileType.BALANCED: self._generate_balanced_profile(base_config),
        }
    
    def _generate_performance_profile(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate performance profile - maximum throughput.
        
        Strategy:
        - FP8 KV cache (50% memory savings)
        - Maximum batch sizes for throughput
        - Aggressive CUDA graphs and compilation
        - All performance optimizations enabled
        - Best attention backend
        """
        config = base_config.copy()
        config["_profile"] = "performance"
        config["_profile_description"] = "Maximum throughput with minimal quality loss"
        
        # KV Cache: FP8 for memory efficiency
        if self.supports_fp8 and self._flag_exists("kv_cache_dtype"):
            choices = self._get_flag_choices("kv_cache_dtype")
            if choices and "fp8" in choices:
                config["kv_cache_dtype"] = "fp8"
            elif choices and "fp8_e4m3" in choices:
                config["kv_cache_dtype"] = "fp8_e4m3"
        
        # Maximum batch sizes
        config["max_num_batched_tokens"] = 65536
        config["max_num_seqs"] = 256
        
        # Aggressive memory utilization
        config["gpu_memory_utilization"] = 0.95
        
        # All standard optimizations (shared across profiles)
        config["enable_chunked_prefill"] = True
        config["enable_prefix_caching"] = True
        config["enforce_eager"] = False
        
        # CUDA graph optimization
        if self._flag_exists("max_cudagraph_capture_size"):
            config["max_cudagraph_capture_size"] = 128
        
        # Compilation for max speed
        if self._flag_exists("compilation_config"):
            config["compilation_config"] = '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
        
        # Expert parallelism for MoE models
        if self._flag_exists("enable_expert_parallel"):
            config["enable_expert_parallel"] = True
        
        # Larger block size
        if self._flag_exists("block_size"):
            choices = self._get_flag_choices("block_size")
            if choices:
                valid_sizes = [s for s in [32, 64] if s in choices]
                if valid_sizes:
                    config["block_size"] = max(valid_sizes)
        
        return config
    
    def _generate_quality_profile(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate quality profile - maximum accuracy, conservative settings.
        
        Strategy:
        - Full precision KV cache (no FP8)
        - Conservative batch sizes
        - Standard block size (16)
        - Disable any aggressive optimizations
        - Prioritize numerical stability
        """
        config = base_config.copy()
        config["_profile"] = "quality"
        config["_profile_description"] = "Maximum accuracy, conservative settings, no quality trade-offs"
        
        # KV Cache: Full precision
        if self._flag_exists("kv_cache_dtype"):
            choices = self._get_flag_choices("kv_cache_dtype")
            if choices and "auto" in choices:
                config["kv_cache_dtype"] = "auto"
            # If no auto, leave as default (don't set)
            elif "kv_cache_dtype" in config:
                del config["kv_cache_dtype"]
        
        # Conservative batch sizes (use base or reduce)
        config["max_num_batched_tokens"] = min(
            base_config.get("max_num_batched_tokens", 32768),
            32768
        )
        config["max_num_seqs"] = min(
            base_config.get("max_num_seqs", 128),
            128
        )
        
        # Enable chunked prefill for stability with long contexts
        config["enable_chunked_prefill"] = True
        
        # Standard block size for quality
        if self._flag_exists("block_size"):
            choices = self._get_flag_choices("block_size")
            if choices and 16 in choices:
                config["block_size"] = 16
        
        # CUDA graphs for consistency
        config["enforce_eager"] = False
        
        # Dtype: Prefer bfloat16 for stability if available
        if self._flag_exists("dtype"):
            choices = self._get_flag_choices("dtype")
            if choices and "bfloat16" in choices:
                config["dtype"] = "bfloat16"
        
        # Conservative memory utilization
        config["gpu_memory_utilization"] = min(
            base_config.get("gpu_memory_utilization", 0.9),
            0.90
        )
        
        # Enable prefix caching for repeated prompts (quality of life)
        if self._flag_exists("enable_prefix_caching"):
            config["enable_prefix_caching"] = True
        
        # Preemption mode: swap to preserve computation
        if self._flag_exists("preemption_mode"):
            choices = self._get_flag_choices("preemption_mode")
            if choices and "swap" in choices:
                config["preemption_mode"] = "swap"
        
        return config
    
    def _generate_balanced_profile(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate balanced profile - good performance with minimal quality loss.
        
        Strategy:
        - FP8 KV cache if on capable hardware (good trade-off)
        - Moderate batch sizes
        - Standard optimizations
        - Middle ground on all settings
        """
        config = base_config.copy()
        config["_profile"] = "balanced"
        config["_profile_description"] = "Balanced performance and quality, good for most use cases"
        
        # KV Cache: FP8 if on H100/H200 (good trade-off)
        if self.supports_fp8 and self._flag_exists("kv_cache_dtype"):
            choices = self._get_flag_choices("kv_cache_dtype")
            if choices:
                if "fp8" in choices:
                    config["kv_cache_dtype"] = "fp8"
                elif "fp8_e4m3" in choices:
                    config["kv_cache_dtype"] = "fp8_e4m3"
        
        # Moderate batch sizes (use base)
        config["max_num_batched_tokens"] = base_config.get("max_num_batched_tokens", 32768)
        config["max_num_seqs"] = base_config.get("max_num_seqs", 128)
        
        # Enable chunked prefill
        config["enable_chunked_prefill"] = True
        
        # Standard block size
        if self._flag_exists("block_size"):
            choices = self._get_flag_choices("block_size")
            if choices and 16 in choices:
                config["block_size"] = 16
        
        # CUDA graphs
        config["enforce_eager"] = False
        
        # Dtype: auto (let vLLM decide)
        if self._flag_exists("dtype"):
            choices = self._get_flag_choices("dtype")
            if choices and "auto" in choices:
                config["dtype"] = "auto"
        
        # Standard memory utilization
        config["gpu_memory_utilization"] = base_config.get("gpu_memory_utilization", 0.90)
        
        # Enable prefix caching
        if self._flag_exists("enable_prefix_caching"):
            config["enable_prefix_caching"] = True
        
        return config
    
    def get_profile_comparison(self, profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comparison table of all profiles.
        
        Returns:
            Dict with differences highlighted
        """
        comparison = {
            "profiles": {},
            "key_differences": [],
        }
        
        # Extract key parameters from each profile
        key_params = [
            "kv_cache_dtype",
            "max_num_batched_tokens",
            "max_num_seqs",
            "block_size",
            "dtype",
            "gpu_memory_utilization",
            "enforce_eager",
            "enable_chunked_prefill",
            "enable_prefix_caching",
        ]
        
        for profile_name, config in profiles.items():
            comparison["profiles"][profile_name] = {
                "description": config.get("_profile_description", ""),
                "max_model_len": config.get("max_model_len"),
                "tensor_parallel_size": config.get("tensor_parallel_size"),
                "settings": {k: config.get(k) for k in key_params if k in config},
            }
        
        # Find differences
        for param in key_params:
            values = {name: config.get(param) for name, config in profiles.items()}
            unique_values = set(values.values())
            if len(unique_values) > 1:
                comparison["key_differences"].append({
                    "parameter": param,
                    "values": values,
                })
        
        return comparison
    
    def save_profiles(self, profiles: Dict[str, Dict[str, Any]], output_dir: Optional[Path] = None):
        """
        Save all profiles to JSON files.
        
        Args:
            profiles: Dict of profile configs
            output_dir: Directory to save (defaults to model_dir)
        """
        if output_dir is None:
            output_dir = self.model_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual profiles
        for profile_name, config in profiles.items():
            profile_path = output_dir / f"vllm_config_{profile_name}.json"
            with open(profile_path, "w") as f:
                # Create a clean version without internal metadata
                clean_config = {k: v for k, v in config.items() if not k.startswith("_")}
                json.dump(clean_config, f, indent=2)
        
        # Save comparison
        comparison = self.get_profile_comparison(profiles)
        comparison_path = output_dir / "vllm_profiles_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        return output_dir


# Convenience functions
def generate_profile_configs(
    base_config: Dict[str, Any],
    model_dir: Path,
    gpu_type: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Quick function to generate all profile configs."""
    generator = ProfileConfigGenerator(model_dir, gpu_type)
    return generator.generate_profiles(base_config)


def compare_profiles(profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Quick function to compare profiles."""
    # Use a dummy generator just for comparison
    generator = ProfileConfigGenerator(Path("."))
    return generator.get_profile_comparison(profiles)
