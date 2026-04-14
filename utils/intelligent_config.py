"""
Intelligent Config Generator

Generates optimal vLLM configuration from capacity calculations.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json

from .capacity_calculator import CapacityEstimate


class IntelligentConfigGenerator:
    """
    Generate optimal vLLM config based on theoretical capacity.
    
    Uses capacity calculations to set:
    - max_num_seqs (concurrent users)
    - max_num_batched_tokens (batching efficiency)
    - GPU utilization
    - KV cache dtype
    - Other optimizations
    """
    
    # GPU-specific optimizations
    GPU_OPTIMIZATIONS = {
        "H200": {
            "max_batched_tokens": 524288,
            "prefers_chunked_prefill": True,
            "prefers_fp8": True,
        },
        "H100": {
            "max_batched_tokens": 524288,
            "prefers_chunked_prefill": True,
            "prefers_fp8": True,
        },
        "A100": {
            "max_batched_tokens": 262144,
            "prefers_chunked_prefill": True,
            "prefers_fp8": False,
        },
    }
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else None
        self.reasoning_log = []
    
    def log_reasoning(self, message: str):
        """Log reasoning step."""
        self.reasoning_log.append(message)
    
    def generate(
        self,
        capacity: CapacityEstimate,
        trust_remote_code: bool = True,
        attention_backend: str = "FLASHINFER",
    ) -> Dict[str, Any]:
        """
        Generate optimal config from capacity estimate.
        
        Args:
            capacity: CapacityEstimate from calculator
            trust_remote_code: Whether to trust remote code
            attention_backend: Attention backend to use
        
        Returns:
            Dict with vLLM configuration
        """
        self.reasoning_log = []
        
        self.log_reasoning(f"Generating config for {capacity.max_concurrent_users} concurrent users")
        self.log_reasoning(f"Target context: {capacity.max_model_len:,} tokens")
        
        # Get GPU-specific optimizations
        gpu_opts = self._get_gpu_opts(capacity)
        
        # Calculate batch tokens for prefill batching
        # max_num_batched_tokens controls how many tokens can be processed in prefill together
        # With good GPUs (H200) and dynamic eviction, we can be more aggressive
        
        # For 32 users with H200:
        # - Total KV capacity: ~1.1M tokens (FP8 across 8 GPUs)
        # - At 60% context (157K): ~7 users worth without eviction
        # - With 32 users: vLLM will evict, but we want to minimize eviction overhead
        # 
        # Strategy: Allow batching that fits comfortably in memory without thrashing
        # If we batch 32 users × 4096 input = 131072 tokens, that's ~12% of KV cache
        # Leaves plenty of room for generation without excessive eviction
        
        avg_input_length = 4096  # Reasonable for thinking model
        
        # Calculate based on user count
        batch_tokens = capacity.max_concurrent_users * avg_input_length
        
        # Cap at GPU-efficient limits (H200 can handle 128K-256K comfortably)
        batch_tokens = min(batch_tokens, gpu_opts["max_batched_tokens"])
        
        # Ensure minimum for efficiency (don't go below 16K for H200)
        batch_tokens = max(batch_tokens, 16384)
        
        self.log_reasoning(f"Calculated max_num_batched_tokens: {batch_tokens:,}")
        self.log_reasoning(f"  = {capacity.max_concurrent_users} users × {avg_input_length:,} input tokens")
        self.log_reasoning(f"  = ~{batch_tokens / capacity.total_kv_capacity_tokens:.1%} of total KV cache")
        self.log_reasoning(f"  Leaves headroom for generation without excessive eviction")
        
        # Build config
        config = {
            # Core capacity (calculated)
            "max_model_len": capacity.max_model_len,
            "max_num_seqs": capacity.max_concurrent_users,
            "max_num_batched_tokens": batch_tokens,
            "tensor_parallel_size": capacity.tensor_parallel_size,
            
            # Memory optimization
            "kv_cache_dtype": capacity.kv_cache_dtype,
            "gpu_memory_utilization": capacity.recommended_gpu_util,
            
            # Performance optimizations
            "enable_chunked_prefill": gpu_opts["prefers_chunked_prefill"],
            "enable_prefix_caching": True,  # Generally beneficial
            "enforce_eager": False,  # Use CUDA graphs
            
            # Model-specific
            "trust_remote_code": trust_remote_code,
            "dtype": "auto",  # Let vLLM decide
            
            # Backend
            "attention_backend": attention_backend,
            
            # MoE-specific
            **({"enable_expert_parallel": True} if capacity.is_moe else {}),
        }
        
        self.log_reasoning(f"Config generated:")
        self.log_reasoning(f"  max_model_len: {config['max_model_len']:,}")
        self.log_reasoning(f"  max_num_seqs: {config['max_num_seqs']}")
        self.log_reasoning(f"  max_num_batched_tokens: {config['max_num_batched_tokens']:,}")
        self.log_reasoning(f"  kv_cache_dtype: {config['kv_cache_dtype']}")
        
        return config
    
    def _get_gpu_opts(self, capacity: CapacityEstimate) -> Dict[str, Any]:
        """Get GPU-specific optimizations."""
        # Check GPU type first
        gpu_type_upper = str(capacity.gpu_memory_gb).upper()  # Use as fallback
        
        # Try to match GPU type
        for gpu_key, opts in self.GPU_OPTIMIZATIONS.items():
            # Check by memory size for H200 (141GB)
            if gpu_key == "H200" and capacity.gpu_memory_gb >= 140:
                return opts
            elif gpu_key == "H100" and 79 <= capacity.gpu_memory_gb <= 81:
                return opts
            elif gpu_key == "A100" and 79 <= capacity.gpu_memory_gb <= 81:
                return opts
        
        # Default for unknown GPUs
        return {
            "max_batched_tokens": 131072,
            "prefers_chunked_prefill": True,
            "prefers_fp8": False,
        }
    
    def get_explanation(self) -> str:
        """Get human-readable explanation of config generation."""
        return "\n".join(self.reasoning_log)
    
    def create_variants(
        self,
        base_config: Dict[str, Any],
        capacity: CapacityEstimate,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create 3 variants from base config:
        - balanced: Base config (optimal for most cases)
        - low_latency: Minimize TTFT
        - high_throughput: Maximize total tok/s
        """
        self.log_reasoning(f"Creating 3 variants from base config")
        
        variants = {
            "balanced": base_config.copy(),
            "low_latency": self._create_low_latency_variant(base_config, capacity),
            "high_throughput": self._create_high_throughput_variant(base_config, capacity),
        }
        
        return variants
    
    def _create_low_latency_variant(
        self,
        base_config: Dict[str, Any],
        capacity: CapacityEstimate,
    ) -> Dict[str, Any]:
        """
        Create low-latency variant:
        - Smaller batching (faster single-request response)
        - Consider disabling chunked prefill
        - Focus on TTFT (time to first token)
        """
        config = base_config.copy()
        
        # Reduce batching for lower latency
        # Use 1x user context instead of 2x
        total_user_context = capacity.max_concurrent_users * capacity.avg_context_per_user
        config["max_num_batched_tokens"] = min(
            total_user_context,  # 1x for lower latency
            131072  # Cap for safety
        )
        
        # Try without chunked prefill (may improve single-request latency)
        # But only if we have enough memory headroom
        if capacity.memory_breakdown.get("safe_kv", 0) > 50:
            config["enable_chunked_prefill"] = False
        
        self.log_reasoning(f"Low-latency variant:")
        self.log_reasoning(f"  max_num_batched_tokens: {config['max_num_batched_tokens']:,} (reduced)")
        self.log_reasoning(f"  enable_chunked_prefill: {config['enable_chunked_prefill']}")
        
        return config
    
    def _create_high_throughput_variant(
        self,
        base_config: Dict[str, Any],
        capacity: CapacityEstimate,
    ) -> Dict[str, Any]:
        """
        Create high-throughput variant:
        - Aggressive batching
        - Ensure chunked prefill is enabled
        - Focus on total tok/s across all users
        """
        config = base_config.copy()
        
        # Increase batching for higher throughput
        # Use 4x user context for more batching
        total_user_context = capacity.max_concurrent_users * capacity.avg_context_per_user
        config["max_num_batched_tokens"] = min(
            total_user_context * 4,  # 4x for aggressive batching
            1048576  # Cap at 1M tokens
        )
        
        # Ensure chunked prefill for throughput
        config["enable_chunked_prefill"] = True
        
        self.log_reasoning(f"High-throughput variant:")
        self.log_reasoning(f"  max_num_batched_tokens: {config['max_num_batched_tokens']:,} (increased)")
        self.log_reasoning(f"  enable_chunked_prefill: True (forced)")
        
        return config


# Convenience functions
def generate_config(capacity: CapacityEstimate, **kwargs) -> Dict[str, Any]:
    """Quick function to generate config."""
    generator = IntelligentConfigGenerator()
    return generator.generate(capacity, **kwargs)


def generate_variants(base_config: Dict[str, Any], capacity: CapacityEstimate) -> Dict[str, Dict[str, Any]]:
    """Quick function to generate variants."""
    generator = IntelligentConfigGenerator()
    return generator.create_variants(base_config, capacity)
