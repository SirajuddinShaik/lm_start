"""
Theoretical Calculator for vLLM Configuration Optimization

Provides memory calculations, performance bounds, and constraint satisfaction
for determining optimal vLLM configurations.

Core philosophy: Calculate theoretical limits, then let empiricism decide.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class HardwareConfig:
    """Hardware configuration for calculations."""

    gpu_count: int
    gpu_memory_gb: float
    gpu_name: str = "unknown"
    gpu_tflops: float = 0.0  # Tensor FP16 TFLOPS
    memory_bandwidth_gb_s: float = 0.0

    # H200 cluster defaults
    @classmethod
    def h200_cluster(cls, gpu_count: int = 8) -> "HardwareConfig":
        return cls(
            gpu_count=gpu_count,
            gpu_memory_gb=141.0,  # H200 has ~141GB usable
            gpu_name="H200",
            gpu_tflops=989.0,  # Tensor FP16
            memory_bandwidth_gb_s=4900.0,  # ~4.9 TB/s
        )

    @classmethod
    def h100_cluster(cls, gpu_count: int = 8) -> "HardwareConfig":
        return cls(
            gpu_count=gpu_count,
            gpu_memory_gb=80.0,
            gpu_name="H100",
            gpu_tflops=989.0,
            memory_bandwidth_gb_s=3350.0,
        )

    @classmethod
    def a100_80gb_cluster(cls, gpu_count: int = 8) -> "HardwareConfig":
        return cls(
            gpu_count=gpu_count,
            gpu_memory_gb=80.0,
            gpu_name="A100-80GB",
            gpu_tflops=312.0,
            memory_bandwidth_gb_s=2039.0,
        )


@dataclass
class ModelConfig:
    """Model configuration for calculations."""

    model_id: str
    model_params_b: float  # Billions of parameters
    max_position_embeddings: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    quantization: Optional[str] = None
    is_moe: bool = False
    is_reasoning: bool = False
    is_multimodal: bool = False

    @property
    def head_dim(self) -> int:
        """Calculate head dimension."""
        return self.hidden_size // self.num_attention_heads

    @property
    def effective_params_b(self) -> float:
        """Get effective parameters after quantization."""
        quant_factors = {
            None: 1.0,
            "fp16": 1.0,
            "bf16": 1.0,
            "fp8": 0.5,
            "fp8_e4m3": 0.5,
            "fp8_e5m2": 0.5,
            "int8": 0.5,
            "int4": 0.25,
            "awq": 0.25,
            "gptq": 0.25,
            "awq_marlin": 0.25,
            "gptq_marlin": 0.25,
            "bitsandbytes": 0.5,
        }
        factor = quant_factors.get(self.quantization, 1.0)
        # MoE models have overhead
        if self.is_moe:
            factor *= 1.3  # Expert routing overhead
        return self.model_params_b * factor


@dataclass
class MemoryRequirements:
    """Calculated memory requirements."""

    model_weights_gb: float
    kv_cache_per_token_gb: float
    activation_memory_gb: float
    overhead_gb: float
    total_required_gb: float

    def for_context(self, context_length: int, batch_size: int = 1) -> float:
        """Calculate memory for given context and batch size."""
        kv_cache = self.kv_cache_per_token_gb * context_length * batch_size
        return (
            self.model_weights_gb
            + kv_cache
            + self.activation_memory_gb
            + self.overhead_gb
        )


@dataclass
class TheoreticalConfig:
    """Theoretically optimal configuration."""

    tensor_parallel_size: int
    pipeline_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    kv_cache_dtype: str
    dtype: str
    required_flags: List[str]
    estimated_memory_gb: float
    estimated_throughput_tok_s: float

    def to_vllm_flags(self) -> Dict[str, Any]:
        """Convert to vLLM flag dictionary."""
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "kv_cache_dtype": self.kv_cache_dtype,
            "dtype": self.dtype,
        }


class TheoreticalCalculator:
    """
    Calculate theoretical vLLM configuration parameters.

    Uses model architecture and hardware specs to estimate:
    - Required tensor parallelism
    - Maximum achievable context length
    - Memory requirements
    - Performance bounds
    """

    def __init__(self, hardware: HardwareConfig):
        self.hardware = hardware
        self.flag_rules = self._load_flag_rules()

    def _load_flag_rules(self) -> Dict:
        """Load flag rules from config."""
        config_path = Path(__file__).parent.parent / "configs" / "flag_rules.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    def calculate_required_tensor_parallel(
        self, model: ModelConfig, target_gpu_utilization: float = 0.70
    ) -> int:
        """
        Calculate minimum tensor parallel size needed.

        Args:
            model: Model configuration
            target_gpu_utilization: Target GPU memory utilization (leaves headroom for KV cache)

        Returns:
            Minimum tensor parallel size (1, 2, 4, or 8)
        """
        # Model size in GB (2 bytes per parameter for FP16)
        bytes_per_param = 2
        model_size_gb = model.effective_params_b * bytes_per_param

        # Add overhead for activations and working memory (15-25%)
        model_size_gb *= 1.2

        # Per-GPU capacity
        per_gpu_capacity = self.hardware.gpu_memory_gb * target_gpu_utilization

        # Required GPUs
        required_gpus = math.ceil(model_size_gb / per_gpu_capacity)

        # Round up to nearest power of 2 that's <= available GPUs
        valid_tp = [1, 2, 4, 8]
        for tp in valid_tp:
            if tp >= required_gpus:
                return min(tp, self.hardware.gpu_count)
        return self.hardware.gpu_count

    def calculate_kv_cache_memory(
        self,
        model: ModelConfig,
        context_length: int,
        batch_size: int = 1,
        dtype_bytes: int = 2,
    ) -> float:
        """
        Calculate KV cache memory in GB.

        Formula: 2 (K and V) * num_layers * hidden_size * dtype_bytes * context * batch
        """
        # Per-token KV cache
        kv_per_token = 2 * model.num_layers * model.hidden_size * dtype_bytes
        total_bytes = kv_per_token * context_length * batch_size
        return total_bytes / (1024**3)

    def calculate_memory_requirements(
        self, model: ModelConfig, context_length: Optional[int] = None
    ) -> MemoryRequirements:
        """
        Calculate memory requirements for model.

        Args:
            model: Model configuration
            context_length: Target context length (defaults to max_position_embeddings)

        Returns:
            MemoryRequirements with breakdown
        """
        if context_length is None:
            context_length = model.max_position_embeddings

        # Model weights (2 bytes per parameter for FP16)
        model_weights_gb = model.effective_params_b * 2

        # KV cache per token
        kv_per_token_gb = self.calculate_kv_cache_memory(model, 1, 1)

        # Activation memory (rough estimate: batch_size * seq_len * hidden * layers * 4 bytes)
        # Assuming batch_size=1, avg_seq_len=context/4 for prefill
        avg_seq_len = context_length / 4
        activation_memory_gb = (
            1 * avg_seq_len * model.hidden_size * model.num_layers * 4 / (1024**3)
        )

        # Overhead (CUDA context, fragmentation, etc.)
        overhead_gb = 2.0  # ~2GB base overhead

        # Total with some context
        total_with_context = (
            model_weights_gb
            + self.calculate_kv_cache_memory(model, context_length)
            + activation_memory_gb
            + overhead_gb
        )

        return MemoryRequirements(
            model_weights_gb=model_weights_gb,
            kv_cache_per_token_gb=kv_per_token_gb,
            activation_memory_gb=activation_memory_gb,
            overhead_gb=overhead_gb,
            total_required_gb=total_with_context,
        )

    def calculate_max_context_from_memory(
        self, model: ModelConfig, available_memory_gb: float, safety_margin: float = 0.9
    ) -> int:
        """
        Calculate maximum context length that fits in available VRAM.

        Args:
            model: Model configuration
            available_memory_gb: Available GPU memory per device
            safety_margin: Safety margin (0.9 = 90% usable)

        Returns:
            Maximum achievable context length
        """
        usable_memory = available_memory_gb * safety_margin

        # Memory used by model weights
        memory = self.calculate_memory_requirements(model, context_length=1)
        model_weights_gb = memory.model_weights_gb

        # Memory available for KV cache
        kv_cache_memory_gb = usable_memory - model_weights_gb - memory.overhead_gb

        if kv_cache_memory_gb <= 0:
            return 4096  # Conservative fallback

        # Bytes per token in KV cache
        bytes_per_token = memory.kv_cache_per_token_gb * (1024**3)

        # Max tokens
        max_tokens = int((kv_cache_memory_gb * 1024**3) / bytes_per_token)

        # Cap at model's max_position_embeddings
        return min(max_tokens, model.max_position_embeddings)

    def estimate_throughput_bounds(
        self, model: ModelConfig, efficiency_factor: float = 0.3
    ) -> Dict[str, float]:
        """
        Estimate theoretical throughput bounds.

        Returns dict with:
            - compute_bound_tok_per_sec: Compute-limited throughput
            - memory_bound_tok_per_sec: Memory-bandwidth-limited throughput
            - realistic_estimate_tok_per_sec: Practical estimate
        """
        # Compute bound: limited by GPU FLOPs
        # Rough estimate: 2 * params * tokens per forward pass
        flops_per_token = 2 * model.model_params_b * 1e9
        compute_capacity = (
            self.hardware.gpu_tflops
            * 1e12
            * self.hardware.gpu_count
            * efficiency_factor
        )
        compute_bound_throughput = compute_capacity / flops_per_token

        # Memory bound: limited by memory bandwidth
        # Rough estimate: memory bandwidth / model size
        if self.hardware.memory_bandwidth_gb_s > 0:
            memory_bound_throughput = (
                self.hardware.memory_bandwidth_gb_s
                * 1e9
                * self.hardware.gpu_count
                / (model.effective_params_b * 1e9 * 2)  # 2 bytes per param
            )
        else:
            memory_bound_throughput = compute_bound_throughput * 2  # Estimate

        # Realistic is typically 40-60% of compute bound for inference
        realistic = min(compute_bound_throughput, memory_bound_throughput) * 0.5

        return {
            "compute_bound_tok_per_sec": compute_bound_throughput,
            "memory_bound_tok_per_sec": memory_bound_throughput,
            "realistic_estimate_tok_per_sec": realistic,
        }

    def generate_theoretical_config(
        self,
        model: ModelConfig,
        target_context: Optional[int] = None,
        enable_optimizations: bool = True,
    ) -> TheoreticalConfig:
        """
        Generate theoretical optimal configuration.

        Args:
            model: Model configuration
            target_context: Target context length (defaults to model max)
            enable_optimizations: Whether to enable advanced optimizations

        Returns:
            TheoreticalConfig with recommended settings
        """
        # Determine tensor parallelism
        tp_size = self.calculate_required_tensor_parallel(model)

        # Calculate available memory per GPU after TP
        memory_per_gpu = self.hardware.gpu_memory_gb

        # Determine max context
        if target_context is None:
            target_context = model.max_position_embeddings

        # Check if target context fits
        memory = self.calculate_memory_requirements(model, target_context)
        memory_per_tp_gpu = memory.total_required_gb / tp_size

        if memory_per_tp_gpu > memory_per_gpu * 0.95:
            # Reduce context to fit
            max_achievable = self.calculate_max_context_from_memory(
                model, memory_per_gpu * tp_size
            )
            target_context = min(target_context, max_achievable)

        # Determine GPU memory utilization
        # Start high, can be reduced empirically if needed
        gpu_mem_util = 0.95

        # Determine dtype based on quantization and hardware
        if model.quantization == "fp8":
            dtype = "auto"
            kv_cache_dtype = "fp8"
        elif model.quantization in ["awq", "gptq", "int4", "int8"]:
            dtype = "auto"
            kv_cache_dtype = "auto"
        else:
            dtype = "auto"  # Let vLLM decide (usually bfloat16 on Ampere+)
            kv_cache_dtype = "auto"

        # Batch parameters - start aggressive
        max_num_batched_tokens = min(65536, target_context * 2)
        max_num_seqs = 256

        # Required flags based on model type
        required_flags = []
        if model.is_moe:
            required_flags.append("--enable-expert-parallel")

        # Estimate throughput
        throughput = self.estimate_throughput_bounds(model)

        # Recalculate memory for final config
        final_memory = self.calculate_memory_requirements(model, target_context)

        return TheoreticalConfig(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,  # Usually 1 for single-node
            max_model_len=target_context,
            gpu_memory_utilization=gpu_mem_util,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            kv_cache_dtype=kv_cache_dtype,
            dtype=dtype,
            required_flags=required_flags,
            estimated_memory_gb=final_memory.total_required_gb,
            estimated_throughput_tok_s=throughput["realistic_estimate_tok_per_sec"],
        )

    def check_config_feasibility(
        self, model: ModelConfig, config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check if a configuration is theoretically feasible.

        Args:
            model: Model configuration
            config: vLLM configuration dictionary

        Returns:
            (is_feasible, reason)
        """
        tp_size = config.get("tensor_parallel_size", 1)
        max_model_len = config.get("max_model_len", model.max_position_embeddings)
        gpu_mem_util = config.get("gpu_memory_utilization", 0.90)

        # Calculate memory with this config
        memory = self.calculate_memory_requirements(model, max_model_len)
        memory_per_gpu = memory.total_required_gb / tp_size
        available_memory = self.hardware.gpu_memory_gb * gpu_mem_util

        if memory_per_gpu > available_memory:
            shortfall = memory_per_gpu - available_memory
            return False, (
                f"Config requires {memory_per_gpu:.1f}GB per GPU but only "
                f"{available_memory:.1f}GB available ({shortfall:.1f}GB shortfall). "
                f"Consider increasing TP size or reducing context."
            )

        # Check TP is valid
        if tp_size not in [1, 2, 4, 8]:
            return (
                False,
                f"Tensor parallel size {tp_size} not in valid values [1, 2, 4, 8]",
            )

        if tp_size > self.hardware.gpu_count:
            return (
                False,
                f"TP size {tp_size} exceeds available GPUs {self.hardware.gpu_count}",
            )

        return True, "Configuration is theoretically feasible"

    def suggest_config_adjustments(
        self, model: ModelConfig, failed_config: Dict[str, Any], error_type: str = "OOM"
    ) -> List[Dict[str, Any]]:
        """
        Suggest adjustments to a failed configuration.

        Args:
            model: Model configuration
            failed_config: The configuration that failed
            error_type: Type of failure (OOM, timeout, etc.)

        Returns:
            List of suggested configuration adjustments
        """
        suggestions = []

        if error_type == "OOM":
            # Priority 1: Reduce batch tokens
            current_batch = failed_config.get("max_num_batched_tokens", 65536)
            for new_batch in [32768, 16384, 8192, 4096, 2048]:
                if new_batch < current_batch:
                    new_config = failed_config.copy()
                    new_config["max_num_batched_tokens"] = new_batch
                    suggestions.append(
                        {
                            "config": new_config,
                            "reason": f"Reduce max_num_batched_tokens from {current_batch} to {new_batch}",
                        }
                    )

            # Priority 2: Reduce max_seqs
            current_seqs = failed_config.get("max_num_seqs", 256)
            for new_seqs in [128, 64, 32, 16]:
                if new_seqs < current_seqs:
                    new_config = failed_config.copy()
                    new_config["max_num_seqs"] = new_seqs
                    suggestions.append(
                        {
                            "config": new_config,
                            "reason": f"Reduce max_num_seqs from {current_seqs} to {new_seqs}",
                        }
                    )

            # Priority 3: Reduce GPU memory utilization
            current_util = failed_config.get("gpu_memory_utilization", 0.95)
            for new_util in [0.90, 0.85, 0.80, 0.75]:
                if new_util < current_util:
                    new_config = failed_config.copy()
                    new_config["gpu_memory_utilization"] = new_util
                    suggestions.append(
                        {
                            "config": new_config,
                            "reason": f"Reduce gpu_memory_utilization from {current_util} to {new_util}",
                        }
                    )

            # Priority 4: Increase tensor parallelism
            current_tp = failed_config.get("tensor_parallel_size", 1)
            for new_tp in [2, 4, 8]:
                if new_tp > current_tp and new_tp <= self.hardware.gpu_count:
                    new_config = failed_config.copy()
                    new_config["tensor_parallel_size"] = new_tp
                    suggestions.append(
                        {
                            "config": new_config,
                            "reason": f"Increase tensor_parallel_size from {current_tp} to {new_tp}",
                        }
                    )

        return suggestions


def get_model_size_from_checkpoint(checkpoint_path: Path) -> Optional[float]:
    """
    Calculate actual model size in billions of parameters from checkpoint files.

    Reads safetensors or pytorch .bin files and sums up all parameter counts.

    Args:
        checkpoint_path: Path to model checkpoint directory

    Returns:
        Parameter count in billions, or None if cannot determine
    """
    if not checkpoint_path or not checkpoint_path.exists():
        return None

    total_params = 0

    # Try safetensors first (preferred format)
    safetensors_files = list(checkpoint_path.glob("*.safetensors"))
    if safetensors_files:
        try:
            from safetensors import safe_open

            for st_file in safetensors_files:
                try:
                    with safe_open(st_file, framework="pt") as f:
                        for key in f.keys():
                            tensor = f.get_tensor(key)
                            total_params += tensor.numel()
                except Exception:
                    continue
            if total_params > 0:
                return total_params / 1e9
        except ImportError:
            pass

    # Try pytorch .bin files
    bin_files = list(checkpoint_path.glob("pytorch_model*.bin")) + \
                list(checkpoint_path.glob("model*.bin"))

    if bin_files:
        try:
            import torch

            for bin_file in bin_files:
                try:
                    state_dict = torch.load(bin_file, map_location="cpu", weights_only=True)
                    for tensor in state_dict.values():
                        if hasattr(tensor, 'numel'):
                            total_params += tensor.numel()
                except Exception:
                    continue
            if total_params > 0:
                return total_params / 1e9
        except Exception:
            pass

    # Try to get from model_index.json or config.json
    config_files = ["model_index.json", "config.json"]
    for cfg_file in config_files:
        cfg_path = checkpoint_path / cfg_file
        if cfg_path.exists():
            try:
                import json
                with open(cfg_path) as f:
                    config = json.load(f)

                # Some configs have explicit parameter count
                if "num_parameters" in config:
                    return config["num_parameters"] / 1e9

                # Calculate from architecture if available
                if "num_hidden_layers" in config and "hidden_size" in config:
                    layers = config["num_hidden_layers"]
                    hidden = config["hidden_size"]
                    vocab = config.get("vocab_size", 50000)
                    intermediate = config.get("intermediate_size", hidden * 4)

                    # Rough estimate: embeddings + layers + head
                    embedding_params = vocab * hidden * 2  # token + position
                    layer_params = layers * (
                        hidden * intermediate * 2 +  # FFN
                        hidden * hidden * 4 +  # Q,K,V,O projections
                        hidden * 2  # layernorms
                    )
                    head_params = hidden * vocab

                    total = embedding_params + layer_params + head_params
                    return total / 1e9

            except Exception:
                continue

    return None


def get_model_size_from_id(model_id: str, checkpoint_path: Optional[Path] = None) -> float:
    """
    Get model size in billions of parameters.

    First tries to calculate from actual checkpoint files (accurate),
    then falls back to parsing model_id (best effort).

    Args:
        model_id: HuggingFace model ID
        checkpoint_path: Optional path to model checkpoint for accurate calculation

    Returns:
        Estimated parameter count in billions
    """
    # First try actual checkpoint calculation
    if checkpoint_path:
        actual_size = get_model_size_from_checkpoint(checkpoint_path)
        if actual_size:
            return actual_size

    # Fallback: parse model_id for common patterns
    model_id_lower = model_id.lower()
    import re

    # Known model mappings (hardcoded for accuracy)
    known_models = {
        "trinity-large": 57.0,  # arcee-ai/Trinity-Large-Thinking-W4A16 = 57B MoE
        "trinity-v1": 32.0,     # Placeholder for Trinity v1 if different
    }
    
    for known_key, known_size in known_models.items():
        if known_key in model_id_lower:
            return known_size

    # Avoid matching quantization patterns like W4A16, W8A8, etc.
    # These are NOT model sizes - they're bit widths for quantization
    # Remove common quantization suffixes before parsing
    cleaned_id = re.sub(r'w\d+a\d+', '', model_id_lower)  # Remove W4A16, W8A8, etc.
    cleaned_id = re.sub(r'awq|gptq|gguf|ggml|fp16|bf16|int8|int4', '', cleaned_id)

    # Extract number before 'b' (e.g., 70B, 7B, 57b)
    # Use word boundary but also require it's not preceded by common quant indicators
    match = re.search(r'(?<![wW])(\d+)(?:b|B)\b', cleaned_id)
    if match:
        size = float(match.group(1))
        # Sanity check: sizes between 0.5B and 1000B are likely valid
        if 0.5 <= size <= 1000:
            return size

    # Check for explicit size mentions
    if "billion" in model_id_lower:
        match = re.search(r'(\d+)\s*billion', model_id_lower)
        if match:
            return float(match.group(1))

    # Cannot determine - return 0 to signal unknown
    return 0.0


def detect_model_capabilities(model_id: str) -> Dict[str, bool]:
    """
    Detect model capabilities from model ID.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Dict with capability flags
    """
    model_id_lower = model_id.lower()

    capabilities = {
        "is_moe": False,
        "is_reasoning": False,
        "is_multimodal": False,
        "supports_tools": False,
        "supports_vision": False,
    }

    # MoE detection
    moe_patterns = ["mixtral", "moe", "dbrx", "deepseek-v3", "trinity"]
    for pattern in moe_patterns:
        if pattern in model_id_lower:
            capabilities["is_moe"] = True
            break

    # Reasoning detection
    reasoning_patterns = ["deepseek-r1", "r1", "qwq", "reasoning", "think", "o1", "o3"]
    for pattern in reasoning_patterns:
        if pattern in model_id_lower:
            capabilities["is_reasoning"] = True
            break

    # Multimodal/vision detection
    vision_patterns = ["vision", "vl", "llava", "pixtral", "gpt-4v", "claude-3-vision"]
    for pattern in vision_patterns:
        if pattern in model_id_lower:
            capabilities["is_multimodal"] = True
            capabilities["supports_vision"] = True
            break

    # Tool support detection
    tool_patterns = [
        "llama-3.",
        "llama3.",  # Llama 3.x has tool support
        "qwen2.5",
        "qwen-2.5",
        "mistral",
        "mixtral",
        "command-r",
        "kimi-k2",
    ]
    for pattern in tool_patterns:
        if pattern in model_id_lower:
            capabilities["supports_tools"] = True
            break

    return capabilities


def calculate_from_model_info(
    model_id: str,
    hardware: Optional[HardwareConfig] = None,
    model_config: Optional[Dict] = None,
) -> TheoreticalConfig:
    """
    Convenience function to calculate config from model info.

    Args:
        model_id: HuggingFace model ID
        hardware: Hardware config (defaults to H200 cluster)
        model_config: Optional model config from HF (with max_position_embeddings, etc.)

    Returns:
        TheoreticalConfig
    """
    if hardware is None:
        hardware = HardwareConfig.h200_cluster()

    # Get model size
    model_params_b = get_model_size_from_id(model_id)

    # Get capabilities
    capabilities = detect_model_capabilities(model_id)

    # Get model architecture from config if available
    if model_config:
        max_pos = model_config.get("max_position_embeddings", 4096)
        num_layers = model_config.get("num_hidden_layers", 32)
        hidden_size = model_config.get("hidden_size", 4096)
        num_heads = model_config.get("num_attention_heads", 32)
        num_kv_heads = model_config.get("num_key_value_heads", num_heads)
    else:
        # Defaults based on size
        if model_params_b >= 70:
            max_pos = 32768
            num_layers = 80
            hidden_size = 8192
            num_heads = 64
            num_kv_heads = 8
        elif model_params_b >= 30:
            max_pos = 32768
            num_layers = 60
            hidden_size = 6656
            num_heads = 52
            num_kv_heads = 8
        else:
            max_pos = 32768
            num_layers = 32
            hidden_size = 4096
            num_heads = 32
            num_kv_heads = 8

    model = ModelConfig(
        model_id=model_id,
        model_params_b=model_params_b,
        max_position_embeddings=max_pos,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        is_moe=capabilities["is_moe"],
        is_reasoning=capabilities["is_reasoning"],
        is_multimodal=capabilities["is_multimodal"],
    )

    calculator = TheoreticalCalculator(hardware)
    return calculator.generate_theoretical_config(model)


# Example usage
if __name__ == "__main__":
    # Example: Calculate config for Llama 3.1 70B on H200 cluster
    hardware = HardwareConfig.h200_cluster(gpu_count=8)

    model = ModelConfig(
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        model_params_b=70.0,
        max_position_embeddings=131072,
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
    )

    calculator = TheoreticalCalculator(hardware)
    config = calculator.generate_theoretical_config(model)

    print(f"Theoretical Config for {model.model_id}:")
    print(f"  Tensor Parallel: {config.tensor_parallel_size}")
    print(f"  Max Model Length: {config.max_model_len:,}")
    print(f"  GPU Memory Util: {config.gpu_memory_utilization}")
    print(f"  Max Batched Tokens: {config.max_num_batched_tokens}")
    print(f"  Max Sequences: {config.max_num_seqs}")
    print(f"  Required Flags: {config.required_flags}")
    print(f"  Estimated Memory: {config.estimated_memory_gb:.1f} GB")
    print(f"  Estimated Throughput: {config.estimated_throughput_tok_s:.0f} tok/s")
