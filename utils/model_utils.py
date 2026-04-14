"""
Model Utilities for lm_start

Provides utilities for working with HuggingFace models including:
- Finding model checkpoints in cache
- Reading model configurations
- Copying documentation and template files
- Calculating model sizes and optimal configurations
"""

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple


def get_hf_home() -> str:
    return os.environ.get("HF_HOME", "/data/.cache/huggingface")


def get_model_cache_path(
    model_id: str, hf_home: Optional[str] = None
) -> Optional[Path]:
    """
    Get the path to a model in the HuggingFace cache.

    Checks both the new HF cache format (hub/models--org--model) and
    the old format (hub/org/model).

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B")
        hf_home: HF_HOME path (defaults to env var or /data/.cache/huggingface)

    Returns:
        Path to model directory or None if not found
    """
    if hf_home is None:
        hf_home = get_hf_home()

    hf_path = Path(hf_home)

    # Try new cache format first
    cache_id = str(model_id).replace("/", "--")
    new_cache = hf_path / "hub" / f"models--{cache_id}"

    if new_cache.exists():
        # Find the snapshot directory
        snapshots = new_cache / "snapshots"
        if snapshots.exists():
            # Get the most recent snapshot
            snapshot_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
            if snapshot_dirs:
                # Sort by modification time
                snapshot_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return snapshot_dirs[0]

    # Try old cache format
    old_cache = hf_path / model_id
    if old_cache.exists():
        return old_cache

    return None


def has_model_files(path: Path) -> bool:
    """Check if directory contains model files."""
    if not path.exists():
        return False

    model_files = list(path.glob("*.bin")) + list(path.glob("*.safetensors"))
    return len(model_files) > 0


def read_model_config(model_id: str, hf_home: Optional[str] = None) -> Optional[Dict]:
    """
    Read config.json from model checkpoint.

    Args:
        model_id: HuggingFace model ID
        hf_home: HF_HOME path

    Returns:
        Config dict or None if not found
    """
    cache_path = get_model_cache_path(model_id, hf_home)
    if not cache_path:
        return None

    config_path = cache_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"[model_utils] Warning: Could not read config.json: {e}")

    return None


def get_max_position_embeddings(model_id: str, hf_home: Optional[str] = None) -> int:
    """
    Get max_position_embeddings from model config.

    Args:
        model_id: HuggingFace model ID
        hf_home: HF_HOME path

    Returns:
        max_position_embeddings or 4096 as default
    """
    config = read_model_config(model_id, hf_home)
    if config:
        # Some models have it in text_config (multimodal)
        if "text_config" in config:
            return config["text_config"].get("max_position_embeddings", 4096)
        return config.get("max_position_embeddings", 4096)
    return 4096


def get_model_size_gb(
    model_id: str,
    hf_home: Optional[str] = None,
) -> Optional[float]:
    """
    Calculate model size in GB from checkpoint files.

    Args:
        model_id: HuggingFace model ID
        hf_home: HF_HOME path

    Returns:
        Size in GB or None if cannot determine
    """
    cache_path = get_model_cache_path(model_id, hf_home)
    if not cache_path:
        return None

    # Try to read config for architecture
    config = read_model_config(model_id, hf_home)
    if not config:
        return None

    # Get architecture parameters
    vocab_size = config.get("vocab_size", 50000)
    hidden_size = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    intermediate_size = config.get("intermediate_size", hidden_size * 4)

    # Check for MoE
    text_config = config.get("text_config", config)
    n_routed_experts = text_config.get("n_routed_experts", 0)
    n_shared_experts = text_config.get("n_shared_experts", 0)
    moe_intermediate_size = text_config.get("moe_intermediate_size", intermediate_size)
    num_experts_per_tok = text_config.get("num_experts_per_tok", 0)

    params = 0

    if n_routed_experts > 0:
        # MoE model calculation
        total_experts = n_routed_experts + n_shared_experts
        expert_params_per_layer = (
            3 * hidden_size * moe_intermediate_size * total_experts
        )
        expert_params = expert_params_per_layer * num_layers
        attn_params = 4 * hidden_size * hidden_size * num_layers
        embed_params = vocab_size * hidden_size if vocab_size and hidden_size else 0
        params = expert_params + attn_params + embed_params
    elif hidden_size and num_layers:
        # Dense model estimation
        params = 7 * hidden_size * hidden_size * num_layers
        if vocab_size and hidden_size:
            params += vocab_size * hidden_size

    # Convert to GB (FP16 = 2 bytes per param, add 20% overhead)
    if params > 0:
        size_gb = (params * 2 / 1e9) * 1.2
        return size_gb

    return None


def calculate_optimal_config(
    model_id: str,
    gpu_count: int,
    gpu_memory_gb: int,
    hf_home: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate optimal vLLM configuration based on model and hardware.

    Args:
        model_id: HuggingFace model ID
        gpu_count: Number of available GPUs
        gpu_memory_gb: Memory per GPU in GB
        hf_home: HF_HOME path

    Returns:
        Dict with optimal configuration
    """
    config = read_model_config(model_id, hf_home)
    model_size_gb = get_model_size_gb(model_id, hf_home)
    max_pos_emb = get_max_position_embeddings(model_id, hf_home)

    # Default values
    result = {
        "tensor_parallel_size": 1,
        "max_model_len": max_pos_emb or 4096,
        "gpu_memory_utilization": 0.90,
        "dtype": "auto",
        "trust_remote_code": False,
        "enable_prefix_caching": True,
        "attention_backend": "FLASHINFER",
    }

    if model_size_gb:
        # Calculate required tensor parallelism
        # Assume we want to use 85% of GPU memory
        usable_memory = gpu_memory_gb * 0.85
        required_tp = int((model_size_gb / usable_memory) + 0.999)  # Round up
        required_tp = max(1, min(required_tp, gpu_count))  # Clamp to available GPUs
        result["tensor_parallel_size"] = required_tp

        # Check if model is large enough to warrant attention
        if model_size_gb > gpu_memory_gb * 0.5:
            result["attention_backend"] = "FLASHINFER"

    # Check for MoE
    if config:
        text_config = config.get("text_config", config)
        if text_config.get("n_routed_experts", 0) > 0:
            result["enable_expert_parallel"] = True

        # Check for quantization
        quant_config = config.get("quantization_config", {})
        if quant_config:
            result["quantization"] = quant_config.get("quant_method", "none")

    return result


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
    reasoning_patterns = [
        "deepseek-r1",
        "r1",
        "qwq",
        "reasoning",
        "think",
        "o1",
        "o3",
        "thinking",
    ]
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
        "llama3.",
        "qwen2.5",
        "qwen-2.5",
        "mistral",
        "mixtral",
        "command-r",
        "kimi.*k2",
    ]
    for pattern in tool_patterns:
        if pattern in model_id_lower:
            capabilities["supports_tools"] = True
            break

    return capabilities


def copy_model_docs(
    model_id: str,
    dest_dir: Path,
    hf_home: Optional[str] = None,
) -> List[str]:
    """
    Copy documentation files from model checkpoint to destination.

    Args:
        model_id: HuggingFace model ID
        dest_dir: Destination directory
        hf_home: HF_HOME path

    Returns:
        List of copied file names
    """
    cache_path = get_model_cache_path(model_id, hf_home)
    if not cache_path:
        return []

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    doc_files = [
        "README.md",
        "LICENSE",
        "LICENSE.txt",
        "NOTICE",
        "NOTICE.txt",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
    ]

    copied = []
    for filename in doc_files:
        src = cache_path / filename
        if src.exists():
            dst = dest_dir / filename
            try:
                shutil.copy2(src, dst)
                copied.append(filename)
            except IOError as e:
                print(f"[model_utils] Warning: Could not copy {filename}: {e}")

    return copied


def copy_chat_template(
    model_id: str,
    dest_dir: Path,
    hf_home: Optional[str] = None,
) -> Optional[str]:
    """
    Copy chat template file from model checkpoint to destination.

    Args:
        model_id: HuggingFace model ID
        dest_dir: Destination directory
        hf_home: HF_HOME path

    Returns:
        Name of copied file or None if not found
    """
    cache_path = get_model_cache_path(model_id, hf_home)
    if not cache_path:
        return None

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Look for chat template files
    template_files = [
        "chat_template.jinja",
        "chat_template.json",
        "tokenizer_config.json",  # Often contains chat_template
    ]

    for filename in template_files:
        src = cache_path / filename
        if src.exists():
            dst = dest_dir / filename
            try:
                shutil.copy2(src, dst)
                return filename
            except IOError as e:
                print(f"[model_utils] Warning: Could not copy {filename}: {e}")

    return None


def get_chat_template(model_id: str, hf_home: Optional[str] = None) -> Optional[str]:
    """
    Get the chat template from model checkpoint.

    Args:
        model_id: HuggingFace model ID
        hf_home: HF_HOME path

    Returns:
        Chat template string or None if not found
    """
    cache_path = get_model_cache_path(model_id, hf_home)
    if not cache_path:
        return None

    # Try chat_template.jinja first
    jinja_path = cache_path / "chat_template.jinja"
    if jinja_path.exists():
        try:
            return jinja_path.read_text()
        except IOError:
            pass

    # Try tokenizer_config.json
    tokenizer_config_path = cache_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        try:
            with open(tokenizer_config_path) as f:
                config = json.load(f)
            return config.get("chat_template")
        except (IOError, json.JSONDecodeError):
            pass

    return None


def copy_all_model_files(
    model_id: str,
    dest_dir: Path,
    hf_home: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Copy all relevant model files from checkpoint to destination.

    Args:
        model_id: HuggingFace model ID
        dest_dir: Destination directory
        hf_home: HF_HOME path

    Returns:
        Dict with lists of copied files by category
    """
    results = {
        "docs": [],
        "templates": [],
        "configs": [],
    }

    # Copy documentation files
    results["docs"] = copy_model_docs(model_id, dest_dir, hf_home)

    # Copy chat template
    template = copy_chat_template(model_id, dest_dir, hf_home)
    if template:
        results["templates"].append(template)

    # Copy tokenizer config (contains chat template if not separate)
    cache_path = get_model_cache_path(model_id, hf_home)
    if cache_path:
        for filename in ["tokenizer_config.json", "preprocessor_config.json"]:
            src = cache_path / filename
            if src.exists():
                dst = dest_dir / filename
                try:
                    shutil.copy2(src, dst)
                    results["configs"].append(filename)
                except IOError:
                    pass

    return results


# =============================================================================
# vLLM Extracted Config Loading
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VLLMExtractedConfig:
    """Structured representation of vLLM extracted configuration."""

    success: bool = False
    architecture: str = "unknown"
    model_type: str = "unknown"
    is_supported: bool = False
    is_moe: bool = False
    quantization: Optional[str] = None
    dtype: str = "auto"
    max_position_embeddings: int = 0
    vocab_size: int = 0
    num_hidden_layers: int = 0
    hidden_size: int = 0
    recommended_tp: int = 1
    memory_estimate_gb: float = 0.0
    total_params_b: float = 0.0  # Total parameters in billions
    active_params_b: float = 0.0  # Active parameters per token (for MoE)
    vllm_version: str = "unknown"
    error: Optional[str] = None

    @property
    def requires_tp(self) -> bool:
        """Check if model requires tensor parallelism."""
        return self.recommended_tp > 1

    @property
    def is_quantized(self) -> bool:
        """Check if model uses quantization."""
        return self.quantization is not None and self.quantization != "None"


def load_vllm_extracted_config(model_dir: Path) -> Optional[VLLMExtractedConfig]:
    """
    Load vLLM extracted configuration from model directory.

    Args:
        model_dir: Path to model directory

    Returns:
        VLLMExtractedConfig if file exists and is valid, None otherwise
    """
    config_path = Path(model_dir) / "vllm_extracted_config.json"

    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            data = json.load(f)

        hf_config = data.get("hf_config", {})
        memory = data.get("memory_estimate", {})

        # Get MoE params if available
        moe_params = data.get("moe_params", {})
        total_params = moe_params.get("total_params_b", memory.get("parameters_b", 0.0))
        active_params = moe_params.get(
            "active_params_b", memory.get("active_parameters_b", total_params)
        )

        return VLLMExtractedConfig(
            success=data.get("success", False),
            architecture=data.get("architecture", "unknown"),
            model_type=hf_config.get("model_type", "unknown"),
            is_supported=data.get("is_supported", False),
            is_moe=data.get("is_moe", False),
            quantization=data.get("vllm_config", {}).get("quantization"),
            dtype=data.get("vllm_config", {}).get("dtype", "auto"),
            max_position_embeddings=hf_config.get("max_position_embeddings", 0),
            vocab_size=hf_config.get("vocab_size", 0),
            num_hidden_layers=hf_config.get("num_hidden_layers", 0),
            hidden_size=hf_config.get("hidden_size", 0),
            recommended_tp=data.get("recommended_tp", 1),
            memory_estimate_gb=memory.get(
                "memory_gb", memory.get("memory_gb_fp16", 0.0)
            ),
            total_params_b=total_params,
            active_params_b=active_params,
            vllm_version=data.get("vllm_version", "unknown"),
            error=data.get("error"),
        )
    except Exception as e:
        print(f"[model_utils] Error loading vllm_extracted_config.json: {e}")
        return None
