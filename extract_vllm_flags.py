#!/usr/bin/env python3
"""
Dynamic vLLM Flag Discovery

Extracts all available vLLM configuration flags dynamically from the
model's virtual environment at runtime. This ensures we always have
the correct flags for the specific vLLM version installed.

Usage:
    python extract_vllm_flags.py <model_dir>

Output:
    Saves vllm_available_flags.json to model_dir with:
    - All EngineArgs with their types, defaults, and choices
    - Categorized by memory, performance, batching, etc.
    - Applicable flags based on model characteristics

This runs INSIDE the model's venv to get accurate flag definitions.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


def log(message: str, level: str = "info"):
    """Print log message."""
    prefix = "[extract_vllm_flags]"
    if level == "error":
        print(f"{prefix} ERROR: {message}", file=sys.stderr)
    else:
        print(f"{prefix} {message}")


def extract_engine_args() -> Dict[str, Any]:
    """
    Extract all EngineArgs from vLLM dynamically.

    Returns dict of flag_name -> flag_info
    """
    try:
        from vllm.engine.arg_utils import EngineArgs
    except ImportError as e:
        log(f"Cannot import vLLM EngineArgs: {e}", "error")
        return {}

    # Create parser and add vLLM args
    parser = argparse.ArgumentParser()
    parser = EngineArgs.add_cli_args(parser)

    flags = {}

    for action in parser._actions:
        if not hasattr(action, "dest") or action.dest == "help":
            continue

        # Determine flag type
        flag_type = "str"
        if action.type:
            type_name = getattr(action.type, "__name__", str(action.type))
            if "int" in type_name.lower():
                flag_type = "int"
            elif "float" in type_name.lower():
                flag_type = "float"
            elif "bool" in type_name.lower():
                flag_type = "bool"

        # Check if it's a boolean (store_true/store_false actions)
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            flag_type = "bool"

        flags[action.dest] = {
            "flag": f"--{action.dest.replace('_', '-')}",
            "type": flag_type,
            "default": action.default,
            "choices": list(action.choices) if action.choices else None,
            "help": action.help if action.help else "",
            "required": getattr(action, "required", False),
        }

    return flags


def get_vllm_version() -> str:
    """Get vLLM version."""
    try:
        import vllm

        return getattr(vllm, "__version__", "unknown")
    except:
        return "unknown"


def categorize_flags(flags: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Categorize flags by their purpose using name patterns.

    Categories:
    - model_loading: dtype, quantization, trust_remote_code, etc.
    - memory: gpu_memory_utilization, max_model_len, kv_cache_dtype, etc.
    - parallelism: tensor_parallel_size, pipeline_parallel_size, etc.
    - batching: max_num_batched_tokens, max_num_seqs, etc.
    - performance: enforce_eager, compilation_config, block_size, etc.
    - caching: enable_prefix_caching, prefix_caching_hash_algo, etc.
    - scheduling: preemption_mode, scheduler_delay_factor, etc.
    - speculative: speculative_model, num_speculative_tokens, etc.
    - serving: port, host, api_key, etc.
    """
    categories = {
        "model_loading": [],
        "memory": [],
        "parallelism": [],
        "batching": [],
        "performance": [],
        "caching": [],
        "scheduling": [],
        "speculative": [],
        "serving": [],
        "other": [],
    }

    # Patterns for categorization
    patterns = {
        "model_loading": [
            "dtype",
            "quantization",
            "trust_remote_code",
            "load_format",
            "tokenizer",
            "tokenizer_mode",
            "config_format",
            "model_impl",
            "revision",
            "code_revision",
            "tokenizer_revision",
        ],
        "memory": [
            "gpu_memory_utilization",
            "max_model_len",
            "kv_cache_dtype",
            "swap_space",
            "num_gpu_blocks_override",
            "max_logprobs",
            "enable_sleep_mode",
        ],
        "parallelism": [
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "enable_expert_parallel",
            "distributed_executor_backend",
            "ray_workers_use_nsight",
            "max_parallel_loading_workers",
            "pipeline_parallel_split_points",
            "all2all_backend",
        ],
        "batching": [
            "max_num_batched_tokens",
            "max_num_seqs",
            "enable_chunked_prefill",
            "max_cpu_loras",
            "max_loras",
            "max_lora_rank",
        ],
        "performance": [
            "enforce_eager",
            "compilation_config",
            "block_size",
            "disable_custom_all_reduce",
            "enable_auto_tool_choice",
            "use_tqdm_on_load",
            "disable_cascade_attn",
            "disable_sliding_window",
            "override_attention_dtype",
            "logits_processors",
        ],
        "caching": [
            "enable_prefix_caching",
            "prefix_caching_hash_algo",
            "mm_cache_preprocessor",
        ],
        "scheduling": [
            "preemption_mode",
            "scheduler_delay_factor",
            "num_lookahead_slots",
            "enable_return_routed_experts",
        ],
        "speculative": [
            "speculative_model",
            "num_speculative_tokens",
            "speculative_max_model_len",
            "speculative_disable_by_batch_size",
            "draft_model",
        ],
        "serving": [
            "port",
            "host",
            "api_key",
            "served_model_name",
            "enable_prompt_embeds",
            "skip_tokenizer_init",
            "allowed_local_media_path",
            "allowed_media_domains",
        ],
    }

    # Categorize each flag
    for flag_name in flags:
        categorized = False
        for category, keywords in patterns.items():
            if any(kw in flag_name.lower() for kw in keywords):
                categories[category].append(flag_name)
                categorized = True
                break

        if not categorized:
            categories["other"].append(flag_name)

    return categories


def get_oom_relevant_flags(flags: Dict[str, Any]) -> Dict[str, str]:
    """
    Identify flags relevant for OOM mitigation with strategies.
    """
    oom_flags = {
        # Memory reduction flags
        "gpu_memory_utilization": "Reduce to 0.85, 0.80 for more headroom",
        "max_model_len": "Reduce context length (last resort)",
        "kv_cache_dtype": "Use 'fp8' on H100/H200 for 50% KV cache reduction",
        "swap_space": "Increase CPU swap space for preemption",
        "num_gpu_blocks_override": "Manually limit KV cache blocks",
        # Batching flags
        "max_num_batched_tokens": "Reduce to 32768, 16384, 8192, 4096",
        "max_num_seqs": "Reduce to 128, 64, 32 for lower concurrency",
        "enable_chunked_prefill": "Enable for better memory efficiency",
        "max_num_partial_prefills": "Reduce parallel prefills",
        # Performance trade-offs
        "enforce_eager": "Disable CUDA graphs (uses less memory)",
        "block_size": "Adjust (16, 32, 64) for memory efficiency",
        "preemption_mode": "Use 'swap' to offload to CPU",
        "disable_custom_all_reduce": "Disable if causing memory issues",
        # Parallelism
        "tensor_parallel_size": "Increase to distribute across more GPUs",
        "pipeline_parallel_size": "Use pipeline parallelism for very large models",
        "enable_expert_parallel": "Enable for MoE models",
        # Speculative (can save memory if draft is smaller)
        "speculative_model": "Use smaller draft model",
    }

    # Filter to only flags that exist
    return {k: v for k, v in oom_flags.items() if k in flags}


def get_profile_recommendations(
    flags: Dict[str, Any], profile: str, gpu_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get flag recommendations for a specific optimization profile.

    Profiles:
    - quality: Maximum accuracy, conservative settings
    - balanced: Good balance of quality and speed
    - speed: Maximum throughput, aggressive optimizations
    """
    recommendations = {}

    if profile == "quality":
        recommendations = {
            "dtype": {"value": "bfloat16", "reason": "Best numerical stability"},
            "enforce_eager": {"value": False, "reason": "CUDA graphs for consistency"},
            "block_size": {"value": 16, "reason": "Optimal for quality"},
            "preemption_mode": {"value": "swap", "reason": "Preserve computation"},
            "kv_cache_dtype": {"value": "auto", "reason": "Full precision cache"},
        }

    elif profile == "balanced":
        recommendations = {
            "dtype": {"value": "auto", "reason": "Let vLLM decide"},
            "enforce_eager": {"value": False, "reason": "Use CUDA graphs"},
            "block_size": {"value": 16, "reason": "Standard size"},
            "enable_chunked_prefill": {"value": True, "reason": "Better throughput"},
        }
        # Add FP8 if on H100/H200
        if gpu_type and any(g in gpu_type.upper() for g in ["H100", "H200"]):
            if "kv_cache_dtype" in flags:
                recommendations["kv_cache_dtype"] = {
                    "value": "fp8",
                    "reason": "50% memory savings with minimal quality loss",
                }

    elif profile == "speed":
        recommendations = {
            "dtype": {"value": "auto", "reason": "Fastest supported"},
            "enforce_eager": {"value": False, "reason": "CUDA graphs essential"},
            "block_size": {"value": 32, "reason": "Larger blocks = less fragmentation"},
            "enable_chunked_prefill": {"value": True, "reason": "Max throughput"},
            "compilation_config": {
                "value": '{"cudagraph_mode": "FULL_AND_PIECEWISE"}',
                "reason": "Torch.compile for max speed",
            },
        }
        # Add FP8 if on H100/H200
        if gpu_type and any(g in gpu_type.upper() for g in ["H100", "H200"]):
            if "kv_cache_dtype" in flags:
                recommendations["kv_cache_dtype"] = {
                    "value": "fp8",
                    "reason": "50% memory savings, faster attention",
                }

    # Filter to only available flags
    return {k: v for k, v in recommendations.items() if k in flags}


def main():
    parser = argparse.ArgumentParser(description="Extract vLLM flags dynamically")
    parser.add_argument("model_dir", help="Model directory")
    parser.add_argument("--gpu-type", default=None, help="GPU type (H200, A100, etc.)")
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=["quality", "balanced", "speed"],
        help="Optimization profile",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    log(f"Extracting vLLM flags for {model_dir.name}")
    log(f"vLLM version: {get_vllm_version()}")

    # Extract all EngineArgs
    flags = extract_engine_args()
    log(f"Found {len(flags)} vLLM flags")

    # Categorize
    categories = categorize_flags(flags)

    # Get OOM-relevant flags
    oom_flags = get_oom_relevant_flags(flags)

    # Get profile recommendations
    profile_recs = get_profile_recommendations(flags, args.profile, args.gpu_type)

    # Build output
    output = {
        "vllm_version": get_vllm_version(),
        "extraction_time": __import__("datetime").datetime.now().isoformat(),
        "flags": flags,
        "categories": categories,
        "oom_relevant_flags": oom_flags,
        "profile_recommendations": {args.profile: profile_recs},
    }

    output_path = (
        model_dir / ".llm-context" / "model-context" / "vllm_available_flags.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log(f"Saved {len(flags)} flags to {output_path}")

    # Print summary
    log("\nFlag Categories:")
    for cat, flag_list in categories.items():
        if flag_list:
            log(f"  {cat}: {len(flag_list)} flags")

    log(f"\nOOM-relevant flags: {len(oom_flags)}")
    log(f"Profile recommendations ({args.profile}): {len(profile_recs)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
