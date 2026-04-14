#!/usr/bin/env python3
"""
Extract vLLM Configuration from Model

This script runs within the model's virtual environment to introspect
vLLM's auto-detected configuration for the model.

Usage:
    python extract_vllm_config.py <model_dir> [--max-model-len N]

Output:
    Saves vllm_extracted_config.json to model_dir with:
    - architecture: Detected model architecture
    - quantization: Quantization method (if any)
    - dtype: Auto-detected dtype (bf16/fp16)
    - attention_backend: vLLM's preferred backend
    - is_moe: Whether model uses MoE
    - memory_requirements: Estimated GPU memory needed
    - recommended_tp: Recommended tensor parallel size
    - vllm_version: vLLM version for compatibility
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))
from utils.model_utils import get_hf_home


def log(message: str, level: str = "info"):
    """Print log message."""
    prefix = "[extract_vllm_config]"
    if level == "error":
        print(f"{prefix} ERROR: {message}", file=sys.stderr)
    else:
        print(f"{prefix} {message}")


def get_model_path_from_dir(model_dir: Path) -> Optional[str]:
    """Get model checkpoint path from model directory."""
    # Try to read model_info.json
    model_info_path = model_dir / ".llm-context" / "model-context" / "model_info.json"
    if model_info_path.exists():
        try:
            with open(model_info_path) as f:
                info = json.load(f)
            model_id = info.get("model_id", "")
            if model_id:
                # Find in HF cache
                cache_id = model_id.replace("/", "--")
                hf_home = get_hf_home()

                # Check hub cache
                hub_path = Path(hf_home) / "hub" / f"models--{cache_id}" / "snapshots"
                if hub_path.exists():
                    snapshots = [d for d in hub_path.iterdir() if d.is_dir()]
                    if snapshots:
                        return str(snapshots[0])

                # Check direct path
                direct_path = Path(hf_home) / model_id
                if direct_path.exists():
                    return str(direct_path)
        except Exception as e:
            log(f"Error reading model_info.json: {e}", "error")

    # Try to find any model files in common locations
    possible_paths = [
        model_dir / "model",  # Local download
    ]

    for path in possible_paths:
        if path.exists() and any(path.glob("*.safetensors")):
            return str(path)

    return None


def get_actual_model_size_gb(model_path: str) -> float:
    """Calculate actual model size from safetensors/bin files in GB."""
    total_bytes = 0
    model_dir = Path(model_path)

    # Check for safetensors files
    for pattern in ["*.safetensors", "*.bin", "*.pt", "*.pth"]:
        for file in model_dir.rglob(pattern):
            if file.is_file():
                total_bytes += file.stat().st_size

    # Also check blobs directory for HF cache
    if "snapshots" in str(model_path):
        # We're in a HF cache snapshot, check parent blobs
        hub_path = Path(model_path).parent.parent
        if hub_path.name == "blobs":
            for file in hub_path.glob("*"):
                if file.is_file() and file.stat().st_size > 1_000_000_000:  # > 1GB
                    total_bytes += file.stat().st_size

    return total_bytes / (1024**3)  # Convert to GB


def introspect_model_with_vllm(
    model_path: str,
    max_model_len: int = 1024,
    trust_remote_code: bool = True,
) -> Dict[str, Any]:
    """
    Introspect model using vLLM's config system.

    Args:
        model_path: Path to model checkpoint
        max_model_len: Max sequence length (use small for introspection)
        trust_remote_code: Whether to trust remote code

    Returns:
        Dict with extracted configuration
    """
    extracted = {
        "success": False,
        "model_path": model_path,
        "error": None,
    }

    try:
        # Import vLLM from the venv
        from vllm.config import ModelConfig as VLLMModelConfig
        from vllm.model_executor.models import ModelRegistry
        from vllm.transformers_utils.config import get_config as get_hf_config

        log(f"Introspecting model: {model_path}")

        # Get HF config first
        hf_config = get_hf_config(model_path, trust_remote_code=trust_remote_code)

        extracted["hf_config"] = {
            "architectures": getattr(hf_config, "architectures", []),
            "model_type": getattr(hf_config, "model_type", "unknown"),
            "vocab_size": getattr(hf_config, "vocab_size", 0),
            "num_hidden_layers": getattr(hf_config, "num_hidden_layers", 0),
            "hidden_size": getattr(hf_config, "hidden_size", 0),
            "num_attention_heads": getattr(hf_config, "num_attention_heads", 0),
            "max_position_embeddings": getattr(hf_config, "max_position_embeddings", 0),
        }

        # Check if architecture is supported
        arch = extracted["hf_config"]["architectures"]
        if arch:
            supported = list(ModelRegistry.get_supported_archs())
            arch_name = arch[0] if isinstance(arch, list) else arch
            extracted["is_supported"] = arch_name in supported
            extracted["architecture"] = arch_name
        else:
            extracted["is_supported"] = False
            extracted["architecture"] = "unknown"

        # Try to create vLLM EngineArgs and extract config info
        try:
            from vllm.engine.arg_utils import EngineArgs

            # Create EngineArgs (vLLM 0.19+ way)
            engine_args = EngineArgs(
                model=model_path,
                tokenizer=model_path,
                tokenizer_mode="auto",
                trust_remote_code=trust_remote_code,
                dtype="auto",
                max_model_len=max_model_len,
                quantization=None,  # Auto-detect
                tensor_parallel_size=1,  # Start with 1 for introspection
            )

            extracted["vllm_config"] = {
                "dtype": str(engine_args.dtype),
                "max_model_len": engine_args.max_model_len,
                "quantization": engine_args.quantization,
                "tensor_parallel_size": engine_args.tensor_parallel_size,
                "trust_remote_code": engine_args.trust_remote_code,
                "load_format": engine_args.load_format,
            }

            # Detect MoE from config
            model_type = extracted["hf_config"]["model_type"]
            extracted["is_moe"] = any(
                x in model_type.lower() for x in ["moe", "mixtral"]
            )
            if not extracted["is_moe"] and arch:
                extracted["is_moe"] = any(
                    x in str(arch).lower() for x in ["moe", "mixtral"]
                )

            # Check for MoE specific config
            if hasattr(hf_config, "num_local_experts"):
                extracted["is_moe"] = True
                extracted["moe_config"] = {
                    "num_local_experts": getattr(hf_config, "num_local_experts", 0),
                    "num_experts_per_tok": getattr(hf_config, "num_experts_per_tok", 0),
                }

            # Check for quantization in config
            if (
                hasattr(hf_config, "quantization_config")
                and hf_config.quantization_config
            ):
                extracted["is_quantized"] = True
                extracted["quantization_method"] = hf_config.quantization_config.get(
                    "quant_method", "unknown"
                )

            extracted["success"] = True

        except Exception as e:
            log(f"vLLM config creation failed: {e}", "error")
            extracted["error"] = str(e)
            extracted["vllm_config"] = None

        # Try to estimate memory requirements
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )

            # Some models (like Kimi-K2.5) nest config in text_config
            if hasattr(config, "text_config") and config.text_config is not None:
                text_cfg = config.text_config
                vocab_size = getattr(
                    text_cfg, "vocab_size", getattr(config, "vocab_size", 50000)
                )
                hidden_size = getattr(
                    text_cfg, "hidden_size", getattr(config, "hidden_size", 4096)
                )
                num_layers = getattr(
                    text_cfg,
                    "num_hidden_layers",
                    getattr(config, "num_hidden_layers", 32),
                )
                intermediate_size = getattr(
                    text_cfg,
                    "intermediate_size",
                    getattr(config, "intermediate_size", hidden_size * 4),
                )
                max_position = getattr(
                    text_cfg,
                    "max_position_embeddings",
                    getattr(config, "max_position_embeddings", 8192),
                )
            else:
                # Standard models - use top-level config
                vocab_size = getattr(config, "vocab_size", 50000)
                hidden_size = getattr(config, "hidden_size", 4096)
                num_layers = getattr(config, "num_hidden_layers", 32)
                intermediate_size = getattr(
                    config, "intermediate_size", hidden_size * 4
                )
                max_position = getattr(config, "max_position_embeddings", 8192)

            # Check for MoE
            num_experts = getattr(config, "num_experts", 0)
            num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
            moe_intermediate_size = getattr(
                config, "moe_intermediate_size", intermediate_size
            )
            num_shared_experts = getattr(config, "num_shared_experts", 0)
            num_dense_layers = getattr(config, "num_dense_layers", 0)

            if num_experts > 0 and num_experts_per_tok > 0:
                # MoE model (like DeepSeek, Qwen3-MoE, Trinity)
                # Dense layers (if specified)
                if num_dense_layers > 0:
                    dense_layers = num_dense_layers
                    moe_layers = num_layers - dense_layers
                else:
                    dense_layers = 0
                    moe_layers = num_layers

                # Dense layers use full intermediate_size
                dense_params = 3 * hidden_size * intermediate_size * dense_layers

                # MoE layers: each expert is smaller (moe_intermediate_size)
                # 3 = gate_proj, up_proj, down_proj
                expert_params_per_layer = (
                    3 * hidden_size * moe_intermediate_size * num_experts
                )
                moe_params = expert_params_per_layer * moe_layers

                # Shared experts (always active)
                shared_params = (
                    3
                    * hidden_size
                    * moe_intermediate_size
                    * num_shared_experts
                    * moe_layers
                )

                # Attention (all layers): Q, K, V, O projections
                attn_params = 4 * hidden_size * hidden_size * num_layers

                # Embeddings: token embeddings (vocab * hidden) + output projection (usually shared or separate)
                embed_params = vocab_size * hidden_size
                # Add output projection if not shared (most MoE models don't share)
                if not getattr(config, "tie_word_embeddings", False):
                    embed_params += vocab_size * hidden_size

                total_params = (
                    dense_params
                    + moe_params
                    + shared_params
                    + attn_params
                    + embed_params
                )

                # Active parameters (per token)
                active_expert_params = (
                    3
                    * hidden_size
                    * moe_intermediate_size
                    * num_experts_per_tok
                    * moe_layers
                )
                active_params = (
                    dense_params
                    + active_expert_params
                    + shared_params
                    + attn_params
                    + embed_params
                )

                params = total_params
                extracted["moe_params"] = {
                    "total_experts": num_experts,
                    "experts_per_token": num_experts_per_tok,
                    "dense_layers": dense_layers,
                    "moe_layers": moe_layers,
                    "total_params_b": total_params / 1e9,
                    "active_params_b": active_params / 1e9,
                }
            else:
                # Dense model standard calculation
                params = (
                    vocab_size * hidden_size * 2  # Embeddings (input + output)
                    + num_layers
                    * (
                        hidden_size * hidden_size * 4  # Attention (Q, K, V, O)
                        + hidden_size * intermediate_size * 2  # FFN (gate+up, down)
                    )
                )
                active_params = params  # Dense: all params are active

            # Estimate memory based on quantization
            quant_config = getattr(config, "quantization_config", None)
            quant_method = quant_config.get("quant_method") if quant_config else None

            # Calculate memory for weights based on quantization
            if quant_method == "compressed-tensors" or "W4A16" in str(model_path):
                # W4A16: 4-bit weights (0.5 bytes), 16-bit activations
                # Weights memory
                weight_memory_gb = (params * 0.5) / 1e9
                # Activation memory (rough estimate: ~2x active params for KV cache, etc)
                activation_memory_gb = (
                    (active_params if num_experts > 0 else params) * 2 / 1e9 * 0.5
                )
                memory_gb = (weight_memory_gb + activation_memory_gb) * 1.2
                quant_info = "W4A16 (INT4 weights, 16-bit activations)"
            elif quant_method in ["awq", "gptq", "bitsandbytes"]:
                # 4-bit quantization ~0.5 bytes per param for weights
                memory_gb = (params * 0.5) / 1e9 * 1.2
                quant_info = f"{quant_method} (4-bit)"
            else:
                # FP16 / BF16: 2 bytes per param
                memory_gb = (params * 2) / 1e9 * 1.2
                quant_info = "FP16/BF16"

            actual_size_gb = get_actual_model_size_gb(model_path)

            extracted["memory_estimate"] = {
                "parameters_b": params / 1e9,
                "active_parameters_b": (active_params if num_experts > 0 else params)
                / 1e9,
                "memory_gb": memory_gb,
                "memory_gb_fp16": (params * 2)
                / 1e9
                * 1.2,  # Theoretical if not quantized
                "quantization": quant_info,
                "actual_size_gb": actual_size_gb,
                "calculated_from_params": params > 0,
            }

        except Exception as e:
            log(f"Memory estimation failed: {e}", "error")
            extracted["memory_estimate"] = None

        # Get vLLM version
        try:
            import vllm

            extracted["vllm_version"] = vllm.__version__
        except Exception:
            extracted["vllm_version"] = "unknown"

    except ImportError as e:
        log(f"vLLM not available in venv: {e}", "error")
        extracted["error"] = f"vLLM import failed: {e}"
    except Exception as e:
        log(f"Unexpected error: {e}", "error")
        extracted["error"] = str(e)

    return extracted


def test_model_load(
    model_path: str,
    tensor_parallel_size: int = 1,
    max_model_len: int = 1024,
) -> Dict[str, Any]:
    """
    Test if model can be loaded with vLLM.

    Returns dict with:
    - success: Whether load succeeded
    - error: Error message if failed
    - actual_memory_gb: Actual memory used (if successful)
    - required_tp: Recommended TP size based on OOM
    """
    result = {
        "success": False,
        "error": None,
        "actual_memory_gb": 0,
        "required_tp": 1,
        "auto_detected_backend": None,
    }

    try:
        from vllm import LLM

        log(f"Testing model load with TP={tensor_parallel_size}...")

        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            disable_log_stats=True,
        )

        # If successful, get actual memory usage
        result["success"] = True
        result["required_tp"] = tensor_parallel_size

        # Try to get memory info from vLLM
        if hasattr(llm, "llm_engine"):
            engine = llm.llm_engine
            if hasattr(engine, "model_executor"):
                # Rough estimate - actual implementation varies by vLLM version
                result["actual_memory_gb"] = "loaded_successfully"

        del llm

    except Exception as e:
        error_str = str(e)
        result["error"] = error_str

        # Parse OOM error for memory requirements
        if "CUDA out of memory" in error_str or "OutOfMemoryError" in error_str:
            log(f"OOM with TP={tensor_parallel_size}, need more GPUs")

            # Try to extract memory requirement from error
            import re

            match = re.search(r"(\d+\.?\d*)\s*GiB", error_str)
            if match:
                required_gb = float(match.group(1))
                result["estimated_memory_gb"] = required_gb

                # Calculate required TP (assuming 141GB per H200 GPU)
                gpu_memory = 141
                required_tp = int(
                    (required_gb / (gpu_memory * 0.85)) + 0.999
                )  # Round up
                result["required_tp"] = min(required_tp, 8)  # Max 8 GPUs

                log(
                    f"Estimated memory: {required_gb:.1f}GB, recommended TP: {result['required_tp']}"
                )

        result["success"] = False

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract vLLM config from model")
    parser.add_argument("model_dir", help="Model directory")
    parser.add_argument(
        "--max-model-len", type=int, default=1024, help="Max model length for testing"
    )
    parser.add_argument(
        "--test-load", action="store_true", help="Actually try loading model"
    )
    parser.add_argument(
        "--tp", type=int, default=1, help="Tensor parallel size for test load"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        log(f"Model directory not found: {model_dir}", "error")
        sys.exit(1)

    # Get model path
    model_path = get_model_path_from_dir(model_dir)
    if not model_path:
        log("Could not find model checkpoint", "error")
        sys.exit(1)

    log(f"Found model at: {model_path}")

    # Introspect model
    extracted = introspect_model_with_vllm(
        model_path=model_path,
        max_model_len=args.max_model_len,
    )

    # Optionally test loading
    if args.test_load:
        log("Testing model load...")
        load_test = test_model_load(
            model_path=model_path,
            tensor_parallel_size=args.tp,
            max_model_len=args.max_model_len,
        )
        extracted["load_test"] = load_test

        # If OOM with TP=1, recommend higher TP
        if not load_test["success"] and load_test.get("required_tp", 1) > 1:
            extracted["recommended_tp"] = load_test["required_tp"]

    # Let agent decide TP based on actual testing
    # Previously: calculated tp from memory_gb / (gpu_memory * 0.85)
    # Now: agent should try different TP values and learn from failures
    extracted["recommended_tp"] = None  # Agent must experiment to find optimal TP

    output_path = (
        model_dir / ".llm-context" / "model-context" / "vllm_extracted_config.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(extracted, f, indent=2)

    log(f"Extracted config saved to: {output_path}")

    # Print summary
    if extracted["success"]:
        log("Extraction successful!")
        log(f"  Architecture: {extracted.get('architecture', 'unknown')}")
        log(f"  Supported: {extracted.get('is_supported', False)}")
        log(f"  Is MoE: {extracted.get('is_moe', False)}")
        if extracted.get("memory_estimate"):
            mem_est = extracted["memory_estimate"]
            log(
                f"  Est. Runtime Memory: ~{mem_est['memory_gb']:.1f}GB (approximate, may vary)"
            )
            log(
                f"  Actual Model Size: {mem_est.get('actual_size_gb', 0):.1f}GB on disk"
            )
            log(f"  Parameters: {mem_est['parameters_b']:.1f}B (estimated from config)")
        log(f"  Recommended TP: Let agent experiment (None)")
    else:
        log(f"Extraction failed: {extracted.get('error')}", "error")

    sys.exit(0 if extracted["success"] else 1)


if __name__ == "__main__":
    main()
