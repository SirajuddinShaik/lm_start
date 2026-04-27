#!/usr/bin/env python3
"""Extract ALL vLLM flags and save as comprehensive YAML."""

import re
import yaml
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def run_vllm_help(model_dir: str, section: str = "all") -> str:
    """Run vllm serve --help in model's venv."""
    vllm_bin = Path(model_dir) / ".venv" / "bin" / "vllm"

    cmd = [str(vllm_bin), "serve", f"--help={section}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout if result.returncode == 0 else result.stderr


def parse_flag_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single flag line from vllm help."""
    line = line.strip()

    # Pattern: --flag-name {choice1,choice2} or --flag-name VALUE (metavar, 2+ chars)
    match = re.match(r"(--[\w-]+)(?:\s+\{([^}]+)\}|\s+([A-Z_\[\]\d]{2,}))?", line)
    if not match:
        return None

    flag_cli = match.group(1)
    flag_name = flag_cli.lstrip("-").replace("-", "_")

    choices = None
    if match.group(2):
        choices = [c.strip().strip("'\"") for c in match.group(2).split(",")]

    return {
        "cli_flag": flag_cli,
        "name": flag_name,
        "choices": choices,
        "arg_type": match.group(3) if match.group(3) else None,
    }


def parse_help_output(text: str) -> Dict[str, Dict[str, Any]]:
    """Parse vllm --help output into structured flags."""
    flags = {}
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Check if this is a flag line
        if line.startswith("--"):
            flag_info = parse_flag_line(line)
            if flag_info:
                flag_name = flag_info["name"]

                # Collect help text and default from following lines
                help_lines = []
                default = None
                i += 1

                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line or next_line.startswith("--"):
                        break

                    help_lines.append(next_line)

                    # Extract default value
                    default_match = re.search(r"\(default:\s*([^)]+)\)", next_line)
                    if default_match:
                        default = default_match.group(1).strip()

                    i += 1

                # Infer choices for boolean flags that don't have {a,b,c} syntax
                # Only trigger on explicit True/False defaults, NOT on "None"
                choices = flag_info["choices"]
                if choices is None and default is not None and default.lower() in ("true", "false"):
                    choices = ["true", "false"]

                flags[flag_name] = {
                    "cli_flag": flag_info["cli_flag"],
                    "choices": choices,
                    "default": default,
                    "help": " ".join(help_lines)[:200] if help_lines else "",
                    "arg_type": flag_info["arg_type"],
                }
                continue

        i += 1

    return flags


def categorize_flag(flag_name: str) -> str:
    """Categorize flag based on its name."""
    categories = {
        "model": [
            "model",
            "tokenizer",
            "dtype",
            "revision",
            "quantization",
            "trust_remote_code",
            "max_model_len",
        ],
        "attention": [
            "attention",
            "flash_attn",
            "flashinfer",
            "triton",
            "sliding_window",
        ],
        "parallel": [
            "tensor_parallel",
            "pipeline_parallel",
            "data_parallel",
            "distributed",
            "expert_parallel",
        ],
        "memory": [
            "gpu_memory",
            "kv_cache",
            "block_size",
            "prefix_caching",
            "cpu_offload",
            "enable_chunked_prefill",
        ],
        "batching": ["max_num_seqs", "max_num_batched", "scheduler", "scheduling"],
        "performance": [
            "enforce_eager",
            "cuda_graph",
            "compilation",
            "cudagraph",
            "torch_compile",
        ],
        "serving": ["port", "host", "api_key", "ssl", "uvicorn", "served_model_name"],
        "multimodal": ["mm_processor", "limit_mm", "video", "image", "interleave_mm"],
        "lora": ["lora", "adapter"],
        "misc": [],
    }

    for cat, patterns in categories.items():
        if any(p in flag_name.lower() for p in patterns):
            return cat
    return "misc"


def _enrich_parser_choices(all_flags: Dict[str, Any], model_dir: str) -> None:
    """Inject known choices for free-text parser flags that vLLM help doesn't enumerate."""
    # Flags sourced from ParserManager.list_registered()
    parser_flags = {
        "reasoning_parser": ("vllm.reasoning", "ReasoningParserManager"),
    }
    for flag_name, (module, cls) in parser_flags.items():
        if flag_name not in all_flags:
            continue
        choices = _get_registered_parsers_direct(model_dir, module, cls)
        if choices:
            all_flags[flag_name]["choices"] = choices

    # Flags sourced from Enum members
    enum_flags = {
        "attention_backend": ("vllm.v1.attention.backends.registry", "AttentionBackendEnum"),
    }
    for flag_name, (module, cls) in enum_flags.items():
        if flag_name not in all_flags:
            continue
        choices = _get_enum_members(model_dir, module, cls)
        if choices:
            all_flags[flag_name]["choices"] = choices


def _get_registered_parsers_direct(model_dir: str, module: str, cls: str) -> Optional[List[str]]:
    venv_python = Path(model_dir) / ".venv" / "bin" / "python"
    code = f"""
try:
    from {module} import {cls} as M
    print(','.join(M.list_registered()))
except Exception:
    print('')
"""
    try:
        result = subprocess.run(
            [str(venv_python), "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        names = [n.strip() for n in result.stdout.strip().split(",") if n.strip()]
        return names if names else None
    except Exception:
        return None


def _get_enum_members(model_dir: str, module: str, cls: str) -> Optional[List[str]]:
    venv_python = Path(model_dir) / ".venv" / "bin" / "python"
    code = f"""
try:
    from {module} import {cls} as E
    print(','.join(e.name for e in E))
except Exception:
    print('')
"""
    try:
        result = subprocess.run(
            [str(venv_python), "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        names = [n.strip() for n in result.stdout.strip().split(",") if n.strip()]
        return names if names else None
    except Exception:
        return None


def extract_vllm_flags_func(
    model_dir: str, gpu_type: Optional[str] = None, profile: str = "balanced"
) -> Dict[str, Any]:
    """Extract all vLLM flags and save as comprehensive YAML."""

    logger.info(f"Extracting vLLM flags for {model_dir}")

    # Get full help output
    help_text = run_vllm_help(model_dir, "all")

    # Parse all flags
    all_flags = parse_help_output(help_text)
    logger.info(f"Found {len(all_flags)} flags")

    if not all_flags:
        return {"success": False, "error": "No flags extracted"}

    # Enrich parser flags with real choices from vLLM's manager registries
    _enrich_parser_choices(all_flags, model_dir)

    # Categorize
    categorized = {}
    for flag_name, flag_info in all_flags.items():
        cat = categorize_flag(flag_name)
        if cat not in categorized:
            categorized[cat] = {}
        categorized[cat][flag_name] = flag_info

    # Build comprehensive output
    output = {
        "metadata": {
            "model_directory": str(model_dir),
            "extraction_time": datetime.now().isoformat(),
            "total_flags": len(all_flags),
            "gpu_type": gpu_type,
            "profile": profile,
        },
        "quick_reference": {
            "critical_flags": [
                "model",
                "tokenizer",
                "dtype",
                "max_model_len",
                "tensor_parallel_size",
                "attention_backend",
                "gpu_memory_utilization",
                "kv_cache_dtype",
            ],
            "oom_relevant": [
                "max_model_len",
                "gpu_memory_utilization",
                "max_num_seqs",
                "max_num_batched_tokens",
                "enable_chunked_prefill",
                "kv_cache_dtype",
                "block_size",
            ],
            "performance_flags": [
                "enforce_eager",
                "compilation_config",
                "attention_backend",
                "enable_prefix_caching",
            ],
        },
        # Flat lookup: every flag by name — use this to find choices/defaults quickly
        "all_flags": all_flags,
        # Same data organised by category
        "flags_by_category": categorized,
    }

    # Save as YAML only
    output_path = Path(model_dir) / ".llm-context" / "model-context" / "vllm_flags.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(
            output, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    logger.info(f"✓ Saved to {output_path} ({len(all_flags)} flags)")

    return {
        "success": True,
        "flags_count": len(all_flags),
        "yaml_path": str(output_path),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Model directory")
    parser.add_argument("--gpu-type", default=None)
    parser.add_argument("--profile", default="balanced")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = extract_vllm_flags_func(args.model_dir, args.gpu_type, args.profile)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    main()
