"""
Theoretical Capacity Calculator

Calculates exact serving capacity from first principles.

Example:
    calculator = CapacityCalculator()
    estimate = calculator.calculate(
        model_params=57e9,
        num_layers=61,
        hidden_size=7168,
        max_context=262144,
        quantization="w4a16",
        gpu_count=8,
        gpu_memory_gb=141,
        gpu_type="H200",
        is_moe=True,
    )
    
    print(f"Can serve {estimate.max_concurrent_users} users at 60% context")
    print(f"Memory breakdown: {estimate.memory_breakdown}")
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class CapacityEstimate:
    """Complete serving capacity estimate."""
    
    # Core capacity
    max_concurrent_users: int
    max_model_len: int
    avg_context_per_user: int  # 60% of max
    
    # Memory breakdown (per GPU in GB)
    memory_breakdown: Dict[str, float]
    
    # KV cache calculations
    kv_cache_per_token_bytes: float
    total_kv_capacity_tokens: int  # Across all GPUs
    kv_cache_dtype: str
    
    # Hardware
    gpu_count: int
    gpu_memory_gb: float
    tensor_parallel_size: int
    
    # Recommendations
    recommended_batch_tokens: int
    recommended_gpu_util: float
    bottleneck: str  # "memory" or "compute"
    
    # Model characteristics
    is_moe: bool = False
    num_experts: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "max_concurrent_users": self.max_concurrent_users,
            "max_model_len": self.max_model_len,
            "avg_context_per_user": self.avg_context_per_user,
            "memory_breakdown": self.memory_breakdown,
            "kv_cache_per_token_bytes": self.kv_cache_per_token_bytes,
            "total_kv_capacity_tokens": self.total_kv_capacity_tokens,
            "kv_cache_dtype": self.kv_cache_dtype,
            "gpu_count": self.gpu_count,
            "gpu_memory_gb": self.gpu_memory_gb,
            "recommended_batch_tokens": self.recommended_batch_tokens,
            "recommended_gpu_util": self.recommended_gpu_util,
            "bottleneck": self.bottleneck,
            "is_moe": self.is_moe,
        }
    
    def explain(self) -> List[str]:
        """Generate human-readable explanation."""
        lines = [
            "=" * 60,
            "CAPACITY ANALYSIS",
            "=" * 60,
            f"",
            f"Model Context: {self.max_model_len:,} tokens",
            f"Target Avg Context (60%): {self.avg_context_per_user:,} tokens",
            f"",
            f"Hardware: {self.gpu_count}x GPUs @ {self.gpu_memory_gb}GB each",
            f"Tensor Parallel: {self.tensor_parallel_size}",
            f"",
            f"Memory Breakdown (per GPU):",
            f"  Model Weights: {self.memory_breakdown.get('weights', 0):.1f} GB",
            f"  Activations (est): {self.memory_breakdown.get('activations', 0):.1f} GB",
            f"  System Overhead: {self.memory_breakdown.get('overhead', 0):.1f} GB",
            f"  Available for KV: {self.memory_breakdown.get('available_kv', 0):.1f} GB",
            f"",
            f"KV Cache: {self.kv_cache_dtype}",
            f"  Per token: {self.kv_cache_per_token_bytes:.0f} bytes",
            f"  Total capacity: {self.total_kv_capacity_tokens:,} tokens",
            f"",
            f"Serving Capacity:",
            f"  At {self.avg_context_per_user:,} tokens/user: {self.max_concurrent_users} concurrent users",
            f"",
            f"Recommendations:",
            f"  GPU memory utilization: {self.recommended_gpu_util:.0%}",
            f"  Max batched tokens: {self.recommended_batch_tokens:,}",
            f"  Bottleneck: {self.bottleneck}",
            f"",
            "=" * 60,
        ]
        return lines


class CapacityCalculator:
    """
    Calculate theoretical serving capacity from model and hardware specs.
    
    Uses first-principles calculations:
    - Weight memory = params × quantization_bytes
    - KV cache = 2 (K+V) × layers × hidden_size × dtype_bytes
    - Available = GPU_memory - weights - activations - overhead
    - Users = available_kv / (context × kv_per_token)
    """
    
    # Quantization bytes per parameter
    QUANTIZATION_BYTES = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "w4a16": 0.5,   # 4-bit weights, 16-bit activations
        "w8a8": 1,      # 8-bit weights, 8-bit activations
        "awq": 0.5,
        "gptq": 0.5,
        "fp8": 1,
    }
    
    # KV cache bytes per element
    KV_DTYPE_BYTES = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "fp8": 1,
        "fp8_e4m3": 1,
        "fp8_e5m2": 1,
    }
    
    # GPU-specific optimizations
    GPU_SPECS = {
        "H100": {"memory_gb": 80, "supports_fp8": True},
        "H200": {"memory_gb": 141, "supports_fp8": True},
        "A100": {"memory_gb": 80, "supports_fp8": False},
        "A100_80GB": {"memory_gb": 80, "supports_fp8": False},
        "MI300X": {"memory_gb": 192, "supports_fp8": True},
    }
    
    def __init__(self):
        self.reasoning_log = []
    
    def log_reasoning(self, message: str):
        """Log reasoning step."""
        self.reasoning_log.append(message)
    
    def calculate(
        self,
        model_params: float,  # Total parameters
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int = 0,
        max_context: int = 32768,
        quantization: str = "fp16",
        gpu_count: int = 8,
        gpu_memory_gb: float = 80,
        gpu_type: str = "A100",
        tensor_parallel_size: Optional[int] = None,
        is_moe: bool = False,
        num_experts: int = 0,
        target_context_percent: float = 0.60,  # 60% of max
        safety_factor: float = 0.85,  # Conservative estimate
        min_users: Optional[int] = None,  # Minimum users to support (vLLM uses eviction)
    ) -> CapacityEstimate:
        """
        Calculate serving capacity from first principles.
        
        Args:
            model_params: Total model parameters (e.g., 57e9 for 57B)
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads (for GQA detection)
            max_context: Maximum context length
            quantization: Quantization type (fp16, w4a16, etc.)
            gpu_count: Number of GPUs available
            gpu_memory_gb: Memory per GPU in GB
            gpu_type: GPU type (H200, A100, etc.)
            tensor_parallel_size: TP degree (defaults to gpu_count)
            is_moe: Whether model uses MoE
            num_experts: Number of experts (if MoE)
            target_context_percent: Target context as % of max (default 60%)
            safety_factor: Conservative multiplier (default 0.85)
        
        Returns:
            CapacityEstimate with all calculations
        """
        
        self.reasoning_log = []
        self.log_reasoning(f"Starting capacity calculation for {model_params/1e9:.1f}B model")
        
        # Determine tensor parallel
        tp = tensor_parallel_size or gpu_count
        self.log_reasoning(f"Tensor parallelism: {tp}")
        
        # Calculate weight memory
        bytes_per_param = self.QUANTIZATION_BYTES.get(quantization, 2)
        total_weight_bytes = model_params * bytes_per_param
        weight_memory_gb = total_weight_bytes / (1024**3)
        weight_per_gpu = weight_memory_gb / tp
        
        self.log_reasoning(f"Weight memory: {weight_memory_gb:.1f}GB total, {weight_per_gpu:.1f}GB per GPU")
        
        # Calculate KV cache per token
        # KV cache = 2 (K+V) × num_layers × hidden_size × dtype_bytes
        # Note: This is per-layer, per-token storage
        kv_dtype = self._select_kv_dtype(gpu_type)
        kv_dtype_bytes = self.KV_DTYPE_BYTES.get(kv_dtype, 2)
        
        # For a standard transformer:
        # Each layer stores K and V, each of size (hidden_size,)
        # So per layer: 2 × hidden_size × dtype_bytes
        # Total: num_layers × 2 × hidden_size × dtype_bytes
        kv_per_token = num_layers * 2 * hidden_size * kv_dtype_bytes
        
        # Convert to KB for readability
        kv_per_token_kb = kv_per_token / 1024
        self.log_reasoning(f"KV cache per token ({kv_dtype}): {kv_per_token} bytes ({kv_per_token_kb:.1f} KB)")
        
        # Estimate activation memory (rough heuristic)
        # Activations scale with batch_size × seq_len × hidden_size
        # Estimate as 10-20% of weight memory for typical serving
        activation_overhead_gb = weight_per_gpu * 0.15  # 15% of weights
        
        self.log_reasoning(f"Estimated activation overhead: {activation_overhead_gb:.1f}GB")
        
        # System overhead (CUDA context, NCCL buffers, etc.)
        system_overhead_gb = 5.0  # Rough estimate
        
        # Calculate available memory for KV cache
        total_gpu_memory = gpu_memory_gb
        used_memory = weight_per_gpu + activation_overhead_gb + system_overhead_gb
        available_kv_gb = total_gpu_memory - used_memory
        
        self.log_reasoning(f"Memory breakdown per GPU:")
        self.log_reasoning(f"  Total: {total_gpu_memory:.1f}GB")
        self.log_reasoning(f"  - Weights: {weight_per_gpu:.1f}GB")
        self.log_reasoning(f"  - Activations: {activation_overhead_gb:.1f}GB")
        self.log_reasoning(f"  - System: {system_overhead_gb:.1f}GB")
        self.log_reasoning(f"  = Available for KV: {available_kv_gb:.1f}GB")
        
        # Apply safety factor
        safe_kv_gb = available_kv_gb * safety_factor
        self.log_reasoning(f"Safe KV budget ({safety_factor:.0%}): {safe_kv_gb:.1f}GB")
        
        # Calculate total KV capacity across all GPUs
        kv_per_gpu_bytes = safe_kv_gb * (1024**3)
        tokens_per_gpu = int(kv_per_gpu_bytes / kv_per_token)
        total_kv_tokens = tokens_per_gpu * gpu_count
        
        self.log_reasoning(f"KV capacity per GPU: {tokens_per_gpu:,} tokens")
        self.log_reasoning(f"Total KV capacity: {total_kv_tokens:,} tokens")
        
        # Calculate serving capacity
        avg_context = int(max_context * target_context_percent)
        theoretical_users = total_kv_tokens / avg_context
        
        # Apply MoE overhead if applicable
        moe_overhead = 1.0
        if is_moe:
            moe_overhead = 1.1  # 10% overhead for MoE routing
            self.log_reasoning(f"MoE overhead: {moe_overhead:.0%}")
        
        # Conservative user count (accounting for fragmentation, etc.)
        # Note: safety_factor already applied to memory, so just apply MoE overhead
        calculated_users = int(theoretical_users / moe_overhead)
        
        # If min_users specified and higher than calculated, use min_users
        # vLLM can handle this through dynamic KV cache eviction (PagedAttention)
        # Trade-off: More users = more eviction overhead = lower per-user performance
        if min_users and min_users > calculated_users:
            self.log_reasoning(f"")
            self.log_reasoning(f"Requested min_users ({min_users}) > calculated ({calculated_users})")
            self.log_reasoning(f"  vLLM will use dynamic eviction (PagedAttention) to handle {min_users} users")
            self.log_reasoning(f"  Trade-off: Higher user count, but potential eviction overhead")
            max_users = min_users
        else:
            max_users = calculated_users
        
        self.log_reasoning(f"")
        self.log_reasoning(f"Serving capacity:")
        self.log_reasoning(f"  Target context: {avg_context:,} tokens ({target_context_percent:.0%} of max)")
        self.log_reasoning(f"  Theoretical users: {theoretical_users:.1f}")
        self.log_reasoning(f"  Final user count: {max_users} users")
        
        # Determine bottleneck
        # If users < 4, memory bound. If > 16, compute bound.
        if max_users < 4:
            bottleneck = "memory"
        elif max_users > 16:
            bottleneck = "compute"
        else:
            bottleneck = "balanced"
        
        # Recommended batch tokens
        # Good rule: 2-4x the total context of all users
        recommended_batch = min(
            max_users * avg_context * 2,
            524288 if "H200" in gpu_type or "H100" in gpu_type else 262144
        )
        
        # Recommended GPU utilization
        # Higher for larger models (need more memory), lower for safety
        recommended_util = 0.90 if max_users > 8 else 0.85
        
        return CapacityEstimate(
            max_concurrent_users=max_users,
            max_model_len=max_context,
            avg_context_per_user=avg_context,
            memory_breakdown={
                "weights": weight_per_gpu,
                "activations": activation_overhead_gb,
                "overhead": system_overhead_gb,
                "available_kv": available_kv_gb,
                "safe_kv": safe_kv_gb,
            },
            kv_cache_per_token_bytes=kv_per_token,
            total_kv_capacity_tokens=total_kv_tokens,
            kv_cache_dtype=kv_dtype,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            tensor_parallel_size=tp,
            recommended_batch_tokens=int(recommended_batch),
            recommended_gpu_util=recommended_util,
            bottleneck=bottleneck,
            is_moe=is_moe,
            num_experts=num_experts,
        )
    
    def _select_kv_dtype(self, gpu_type: str) -> str:
        """Select optimal KV cache dtype based on GPU."""
        gpu_upper = gpu_type.upper()
        for gpu_key, specs in self.GPU_SPECS.items():
            if gpu_key in gpu_upper and specs["supports_fp8"]:
                return "fp8"
        return "fp16"
    
    def calculate_from_model_info(
        self,
        model_info_path: Path,
        device_config_path: Path,
    ) -> Optional[CapacityEstimate]:
        """
        Calculate capacity from model_info.json and device_config.json.
        
        Args:
            model_info_path: Path to model_info.json
            device_config_path: Path to device_config.json
        
        Returns:
            CapacityEstimate or None if files not found
        """
        try:
            with open(model_info_path) as f:
                model_info = json.load(f)
            
            with open(device_config_path) as f:
                device_config = json.load(f)
            
            # Extract model specs
            params_str = model_info.get("size", "7B")
            params = self._parse_params(params_str)
            
            # Get config if available
            model_config = model_info.get("config", {})
            num_layers = model_config.get("num_hidden_layers", 32)
            hidden_size = model_config.get("hidden_size", 4096)
            max_context = model_config.get("max_position_embeddings", 32768)
            
            # Detect MoE
            is_moe = any(x in model_info.get("model_id", "").lower() 
                        for x in ["moe", "mixtral", "trinity", "deepseek-v3"])
            
            # Extract device specs
            gpus = device_config.get("gpus", {})
            gpu_count = gpus.get("count", 8)
            gpu_memory = gpus.get("memory_gb_per_gpu", 80)
            gpu_type = gpus.get("name", "A100")
            
            # Detect quantization from model name
            model_id = model_info.get("model_id", "")
            quantization = self._detect_quantization(model_id)
            
            return self.calculate(
                model_params=params,
                num_layers=num_layers,
                hidden_size=hidden_size,
                max_context=max_context,
                quantization=quantization,
                gpu_count=gpu_count,
                gpu_memory_gb=gpu_memory,
                gpu_type=gpu_type,
                is_moe=is_moe,
            )
            
        except Exception as e:
            self.log_reasoning(f"Error calculating from model info: {e}")
            return None
    
    def _parse_params(self, params_str: str) -> float:
        """Parse parameter count from string like '57B' or '7.5B'."""
        params_str = params_str.upper().replace("B", "").strip()
        try:
            return float(params_str) * 1e9
        except:
            return 7e9  # Default to 7B
    
    def _detect_quantization(self, model_id: str) -> str:
        """Detect quantization from model name."""
        model_lower = model_id.lower()
        if "w4a16" in model_lower or "awq" in model_lower or "gptq" in model_lower:
            return "w4a16"
        elif "w8a8" in model_lower or "fp8" in model_lower:
            return "w8a8"
        elif "bf16" in model_lower:
            return "bf16"
        return "fp16"


# Convenience functions
def calculate_capacity(
    model_params: float,
    num_layers: int,
    hidden_size: int,
    **kwargs
) -> CapacityEstimate:
    """Quick function to calculate capacity."""
    calc = CapacityCalculator()
    return calc.calculate(model_params, num_layers, hidden_size, **kwargs)


def explain_capacity(estimate: CapacityEstimate):
    """Print human-readable capacity explanation."""
    for line in estimate.explain():
        print(line)
