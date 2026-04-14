#!/usr/bin/env python3
"""
Generate PM2 ecosystem.config.js for vLLM model serving.

Uses optimized_config.json if available (from optimize_config.py),
otherwise falls back to auto-detection from model_info.json and model config.

Key improvements:
- Uses max_position_embeddings from config.json for max_seq_length
- Calculates optimal tensor parallel based on model size
- Applies model family specific defaults
- Copies documentation files from checkpoint
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from utils.model_utils import (
    get_hf_home,
    get_model_cache_path,
    read_model_config,
    get_max_position_embeddings,
    calculate_optimal_config,
    copy_model_docs,
    copy_chat_template,
    detect_model_capabilities,
    copy_all_model_files,
)
from vllm_flag_validator import validate_vllm_flags


SCRIPT_DIR = Path(__file__).parent
CONFIGS_DIR = SCRIPT_DIR / "configs"


def load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get GPU information from nvidia-smi."""
    gpus = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        vram = int(parts[1].strip())
                        gpus.append({"name": name, "vram_mb": vram})
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return gpus


def load_model_defaults() -> Dict[str, Any]:
    """Load model family specific defaults."""
    defaults_path = CONFIGS_DIR / "model_defaults.json"
    if defaults_path.exists():
        return load_json(str(defaults_path))
    return {"families": {}, "size_based_rules": {}}


def detect_model_family(model_id: str, config: Optional[Dict] = None) -> str:
    """Detect model family from model ID and config."""
    model_id_lower = model_id.lower()

    # Check config for architectures
    if config:
        arch = config.get("architectures", [])
        if arch:
            arch_name = arch[0].lower()
            if "llama" in arch_name:
                return "llama"
            elif "mistral" in arch_name:
                return "mistral"
            elif "mixtral" in arch_name:
                return "mixtral"
            elif "qwen2moe" in arch_name:
                return "qwen-moe"
            elif "qwen" in arch_name:
                return "qwen"
            elif "gemma2" in arch_name:
                return "gemma2"
            elif "gemma" in arch_name:
                return "gemma"
            elif "phi3" in arch_name:
                if "vision" in model_id_lower:
                    return "phi-vl"
                return "phi"

    # Fallback to model ID patterns
    family_patterns = {
        "deepseek-reasoning": ["deepseek-r1", "deepseek-prover"],
        "deepseek": ["deepseek-v2", "deepseek-v3", "deepseek-coder-v2"],
        "kimi": ["kimi"],
        "qwen-moe": ["qwen2-moe", "qwen2.5-moe"],
        "qwen-vl": ["qwen2-vl", "qwen2.5-vl"],
        "qwen": ["qwen"],
        "mixtral": ["mixtral"],
        "mistral": ["mistral"],
        "phi-vl": ["phi-3-vision", "phi3-vision"],
        "phi": ["phi-3", "phi3"],
        "gemma2": ["gemma-2", "gemma2"],
        "gemma": ["gemma"],
        "llama": ["llama"],
    }

    for family, patterns in family_patterns.items():
        for pattern in patterns:
            if pattern in model_id_lower:
                return family

    return "llama"  # Default fallback


def generate_vllm_args(
    model_id: str,
    model_info: Dict[str, Any],
    device_config: Dict[str, Any],
    model_config: Optional[Dict] = None,
    model_dir: Optional[Path] = None,
) -> tuple[List[str], Dict[str, Any]]:
    """
    Generate optimal vLLM arguments.

    Returns:
        Tuple of (args list, environment dict)
    """
    gpu_count = device_config.get("gpus", {}).get("count", 8)
    gpu_memory_gb = device_config.get("gpus", {}).get("memory_gb_per_gpu", 80)

    # Get optimal configuration based on model and hardware
    optimal = calculate_optimal_config(model_id, gpu_count, gpu_memory_gb)

    # Get model family defaults
    defaults = load_model_defaults()
    family = detect_model_family(model_id, model_config)
    family_defaults = defaults.get("families", {}).get(family, {})

    # Determine tensor parallel size
    tensor_parallel = optimal["tensor_parallel_size"]

    # Use max_position_embeddings from config if available
    max_model_len = optimal["max_model_len"]
    if model_config:
        max_pos = get_max_position_embeddings(model_id)
        if max_pos:
            # Use max_position_embeddings directly as requested
            max_model_len = max_pos
            print(f"  Using max_position_embeddings from config: {max_model_len}")

    # Build flag config for validation
    flag_config = {
        "tensor_parallel_size": tensor_parallel,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": optimal["gpu_memory_utilization"],
        "dtype": optimal["dtype"],
    }

    # Add family-specific flags
    if family_defaults.get("trust_remote_code"):
        flag_config["trust_remote_code"] = True

    if family_defaults.get("enable_prefix_caching", True):
        flag_config["enable_prefix_caching"] = True

    if family_defaults.get("enable_expert_parallel") or optimal.get(
        "enable_expert_parallel"
    ):
        flag_config["enable_expert_parallel"] = True

    if family_defaults.get("is_multimodal") or optimal.get("is_multimodal"):
        flag_config["limit_mm_per_prompt"] = "image=5"

    if family_defaults.get("is_reasoning"):
        flag_config["reasoning_parser"] = family_defaults.get(
            "reasoning_parser", "deepseek"
        )

    if family_defaults.get("enable_auto_tool_choice"):
        flag_config["enable_auto_tool_choice"] = True
        flag_config["tool_call_parser"] = family_defaults.get(
            "tool_call_parser", "auto"
        )

    special_args = family_defaults.get("special_args", [])
    i = 0
    while i < len(special_args):
        arg = special_args[i]
        if arg.startswith("--"):
            flag_name = arg.lstrip("-").replace("-", "_")
            if i + 1 < len(special_args) and not special_args[i + 1].startswith("--"):
                value = special_args[i + 1]
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                flag_config[flag_name] = value
                i += 2
            else:
                flag_config[flag_name] = True
                i += 1
        else:
            i += 1

    validation_model_dir = (
        model_dir
        if model_dir
        else Path(device_config.get("paths", {}).get("models_dir", "/models"))
    )
    validated_args = validate_vllm_flags(
        flag_config, model_dir=str(validation_model_dir)
    )

    # Build args (model_id and host/port are not validated flags)
    args = [
        "serve",
        model_id,
        "--host",
        device_config.get("vllm_defaults", {}).get("host", "0.0.0.0"),
        "--port",
        str(device_config.get("vllm_defaults", {}).get("port", 8000)),
    ] + validated_args

    visible_devices = device_config.get("gpus", {}).get(
        "visible_devices", "0,1,2,3,4,5,6,7"
    )
    devices = visible_devices.split(",")[:tensor_parallel]

    env = {
        "HF_HOME": device_config.get("paths", {}).get("hf_home", get_hf_home()),
        "CUDA_VISIBLE_DEVICES": ",".join(devices),
        "VLLM_ATTENTION_BACKEND": family_defaults.get(
            "attention_backend", "FLASHINFER"
        ),
    }

    # Add device environment
    env.update(device_config.get("environment", {}))

    return args, env


def generate_ecosystem_config(
    model_dir: Path,
    model_info: Dict[str, Any],
    device_config: Dict[str, Any],
    model_config: Optional[Dict] = None,
    output_file: Optional[Path] = None,
) -> str:
    """Generate ecosystem.config.js with optimal settings."""

    model_id = model_info.get("model_id", "unknown/model")
    model_name = model_id.split("/")[-1]
    app_name = model_name.lower().replace("-", "_").replace(".", "_")

    # Generate vLLM args and environment
    args, env = generate_vllm_args(
        model_id, model_info, device_config, model_config, model_dir
    )

    # Extract tensor parallel from args
    tensor_parallel = 1
    for i, arg in enumerate(args):
        if arg == "--tensor-parallel-size" and i + 1 < len(args):
            tensor_parallel = int(args[i + 1])
            break

    # Determine vLLM path
    venv_path = model_dir / ".venv"
    if venv_path.exists():
        vllm_path = str(venv_path / "bin" / "vllm")
    else:
        vllm_path = "vllm"

    # PM2 settings
    pm2_config = device_config.get("pm2", {})
    logs_dir = model_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Serialize for JS
    env_json = json.dumps(env, indent=6)
    args_json = json.dumps(args)

    config = f"""module.exports = {{
  apps: [{{
    // Model: {model_id}
    // Generated: {__import__("datetime").datetime.now().isoformat()}
    name: '{app_name}',
    script: '{vllm_path}',
    args: {args_json},
    interpreter: 'none',
    cwd: '{model_dir}',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: 'auto',

    // Environment
    env: {env_json},

    // Logging
    log_date_format: '{pm2_config.get("log_date_format", "YYYY-MM-DD HH:mm:ss Z")}',
    error_file: '{logs_dir}/{app_name}_server.err.log',
    out_file: '{logs_dir}/{app_name}_server.out.log',
    merge_logs: false,

    // Restart policy
    exp_backoff_restart_delay: 100,
    max_restarts: {pm2_config.get("max_restarts", 10)},
    min_uptime: '{pm2_config.get("min_uptime", "10s")}',
    restart_delay: {pm2_config.get("restart_delay", 1000)},

    // Graceful shutdown
    kill_timeout: {pm2_config.get("kill_timeout", 60000)},
    wait_ready: true,
    listen_timeout: {pm2_config.get("listen_timeout", 300000)},
  }}]
}};
"""

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(config)
        print(f"\nConfig written to: {output_file}")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimal PM2 ecosystem.config.js for vLLM serving"
    )
    parser.add_argument("model_dir", help="Model directory containing .llm-context/")
    parser.add_argument(
        "-o",
        "--output",
        default="ecosystem.config.js",
        help="Output file path (default: ecosystem.config.js in model_dir)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--model-info",
        help="Path to model_info.json (default: .llm-context/model-context/model_info.json)",
    )
    parser.add_argument(
        "--device-config",
        help="Path to device config (default: .llm-context/model-context/device_config.json)",
    )
    parser.add_argument(
        "--copy-docs",
        action="store_true",
        help="Copy documentation files from checkpoint",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print config without writing"
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load model info
    model_info_path = (
        Path(args.model_info)
        if args.model_info
        else model_dir / ".llm-context" / "model-context" / "model_info.json"
    )
    if not model_info_path.exists():
        print(f"ERROR: model_info.json not found at {model_info_path}")
        sys.exit(1)

    model_info = load_json(str(model_info_path))
    model_id = model_info.get("model_id", "unknown/model")

    print(f"Model: {model_id}")

    # Load device config
    if args.device_config:
        device_config_path = Path(args.device_config)
    else:
        device_config_path = (
            model_dir / ".llm-context" / "model-context" / "device_config.json"
        )

    if device_config_path.exists():
        device_config = load_json(str(device_config_path))
    else:
        # Fallback to system config
        system_config_path = Path(__file__).parent / "config" / "system.yaml"
        if system_config_path.exists():
            import yaml

            with open(system_config_path) as f:
                config = yaml.safe_load(f)
            device_config = {
                "gpus": config.get("device", {}).get("gpu", {}),
                "paths": config.get("paths", {}),
                "environment": config.get("environment", {}),
                "pm2": config.get("pm2", {}),
                "vllm_defaults": config.get("vllm_defaults", {}),
            }
        else:
            device_config = {
                "gpus": {"count": 8, "memory_gb_per_gpu": 80},
                "paths": {},
                "environment": {},
                "pm2": {},
                "vllm_defaults": {"host": "0.0.0.0", "port": 8000},
            }

    # Read model config from HF cache
    model_config = read_model_config(model_id)
    if model_config:
        print(f"  Loaded model config from checkpoint")
        max_pos = get_max_position_embeddings(model_id)
        if max_pos:
            print(f"  max_position_embeddings: {max_pos}")
    else:
        print(f"  Warning: Could not load model config from checkpoint")

    # Copy documentation files if requested
    if args.copy_docs:
        print("\nCopying files from model checkpoint...")
        copied = copy_all_model_files(model_id, model_dir)

        if copied["docs"]:
            print(f"  Documentation: {', '.join(copied['docs'])}")
        if copied["templates"]:
            print(f"  Templates: {', '.join(copied['templates'])}")
        if copied["configs"]:
            print(f"  Configs: {', '.join(copied['configs'])}")

        if not any(copied.values()):
            print("  No files found to copy")

    # Generate config
    output_path = (
        model_dir / "ecosystem.config.js"
        if args.output == "ecosystem.config.js"
        else Path(args.output)
    )

    print("\nGenerating vLLM configuration...")
    config = generate_ecosystem_config(
        model_dir=model_dir,
        model_info=model_info,
        device_config=device_config,
        model_config=model_config,
        output_file=None if args.dry_run else output_path,
    )

    if args.dry_run:
        print("\n" + "=" * 60)
        print(config)


if __name__ == "__main__":
    main()
