#!/usr/bin/env python3
"""
vLLM Flag Validator - Runtime flag introspection and validation

This module queries vLLM at runtime to determine:
1. Which flags are boolean (no value) vs value-required
2. Validates and formats flags correctly
3. Removes invalid flags and reports them
"""

import subprocess
import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum


class FlagType(Enum):
    BOOLEAN = "boolean"  # --flag or --no-flag (no value)
    VALUE = "value"  # --flag VALUE (requires value)
    CHOICE = "choice"  # --flag {choice1,choice2} (enum)


@dataclass
class FlagInfo:
    name: str
    flag_type: FlagType
    has_negative: bool = False  # Has --no-variant
    choices: Optional[List[str]] = None
    default_value: Optional[str] = None
    description: str = ""


@dataclass
class ValidationResult:
    valid_args: List[str] = field(default_factory=list)
    removed_flags: List[Tuple[str, str]] = field(default_factory=list)  # (flag, reason)
    warnings: List[str] = field(default_factory=list)


class VLLMFlagValidator:
    """Validates and formats vLLM CLI flags by introspecting vLLM at runtime."""

    def __init__(
        self, vllm_binary: Optional[str] = None, model_dir: Optional[str] = None
    ):
        """
        Initialize validator.

        Args:
            vllm_binary: Path to vllm binary. If None, will try to find it.
            model_dir: Model directory (to find .venv/bin/vllm)
        """
        self.vllm_binary = vllm_binary or self._find_vllm_binary(model_dir)
        self._flag_cache: Optional[Dict[str, FlagInfo]] = None
        self._cache_file: Optional[Path] = None

        if model_dir:
            self._cache_file = Path(model_dir) / ".vllm_flag_cache.json"

    def _find_vllm_binary(self, model_dir: Optional[str] = None) -> str:
        """Find vLLM binary."""
        if model_dir:
            venv_vllm = Path(model_dir) / ".venv" / "bin" / "vllm"
            if venv_vllm.exists():
                return str(venv_vllm)

        # Try system vllm
        try:
            result = subprocess.run(
                ["which", "vllm"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            pass

        # Default fallback
        return "vllm"

    def _parse_help_output(self) -> Dict[str, FlagInfo]:
        """Parse vLLM --help=all output to extract flag information."""
        flags = {}

        try:
            result = subprocess.run(
                [self.vllm_binary, "serve", "--help=all"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            help_text = result.stdout
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            print(f"Warning: Could not query vLLM help: {e}", file=sys.stderr)
            return self._get_builtin_flags()

        # Parse flags from help text
        # Pattern 1: Boolean with negative variant: --flag, --no-flag (, -short)
        bool_pattern = re.compile(
            r"^\s+(--[\w-]+),\s+(--no-[\w-]+)(?:,\s+-\w+)?(?:\s|$)", re.MULTILINE
        )

        # Pattern 2: Value flag: --flag VALUE or --flag {choices}
        value_pattern = re.compile(
            r"^\s+(--[\w-]+)\s+([A-Z_]+(?:\s+\[[A-Z_]+\])?|\{[^}]+\})(?:,\s+-\w+\s+\w+)?(?:\s|$)",
            re.MULTILINE,
        )

        # Pattern 3: Standalone boolean (no negative variant)
        standalone_bool_pattern = re.compile(
            r"^\s+(--[\w-]+)(?:\s|$)(?!,\s+--no-)", re.MULTILINE
        )

        # Find all boolean flags with negative variants
        for match in bool_pattern.finditer(help_text):
            positive_flag = match.group(1)
            negative_flag = match.group(2)
            flag_name = positive_flag.lstrip("-").replace("-", "_")

            flags[flag_name] = FlagInfo(
                name=flag_name, flag_type=FlagType.BOOLEAN, has_negative=True
            )

        # Find all value flags
        for match in value_pattern.finditer(help_text):
            flag = match.group(1)
            value_spec = match.group(2)
            flag_name = flag.lstrip("-").replace("-", "_")

            # Skip if already found as boolean
            if flag_name in flags:
                continue

            if value_spec.startswith("{") and value_spec.endswith("}"):
                # It's a choice/enum flag
                choices = [c.strip() for c in value_spec[1:-1].split(",")]
                flags[flag_name] = FlagInfo(
                    name=flag_name, flag_type=FlagType.CHOICE, choices=choices
                )
            else:
                # It's a value flag
                flags[flag_name] = FlagInfo(name=flag_name, flag_type=FlagType.VALUE)

        # Cache if possible
        if self._cache_file:
            try:
                cache_data = {
                    name: {
                        "flag_type": info.flag_type.value,
                        "has_negative": info.has_negative,
                        "choices": info.choices,
                    }
                    for name, info in flags.items()
                }
                with open(self._cache_file, "w") as f:
                    json.dump(cache_data, f)
            except Exception as e:
                print(f"Warning: Could not cache flags: {e}", file=sys.stderr)

        return flags

    def _get_builtin_flags(self) -> Dict[str, FlagInfo]:
        """Return built-in flag definitions as fallback."""
        return {
            # Boolean flags
            "trust_remote_code": FlagInfo(
                "trust_remote_code", FlagType.BOOLEAN, has_negative=True
            ),
            "enforce_eager": FlagInfo(
                "enforce_eager", FlagType.BOOLEAN, has_negative=True
            ),
            "enable_chunked_prefill": FlagInfo(
                "enable_chunked_prefill", FlagType.BOOLEAN, has_negative=True
            ),
            "enable_prefix_caching": FlagInfo(
                "enable_prefix_caching", FlagType.BOOLEAN, has_negative=True
            ),
            "enable_auto_tool_choice": FlagInfo(
                "enable_auto_tool_choice", FlagType.BOOLEAN, has_negative=True
            ),
            "enable_expert_parallel": FlagInfo(
                "enable_expert_parallel", FlagType.BOOLEAN, has_negative=True
            ),
            "disable_log_requests": FlagInfo(
                "disable_log_requests", FlagType.BOOLEAN, has_negative=False
            ),
            "disable_log_stats": FlagInfo(
                "disable_log_stats", FlagType.BOOLEAN, has_negative=False
            ),
            "skip_tokenizer_init": FlagInfo(
                "skip_tokenizer_init", FlagType.BOOLEAN, has_negative=True
            ),
            "enable_sleep_mode": FlagInfo(
                "enable_sleep_mode", FlagType.BOOLEAN, has_negative=True
            ),
            "enable_prompt_embeds": FlagInfo(
                "enable_prompt_embeds", FlagType.BOOLEAN, has_negative=True
            ),
            "disable_sliding_window": FlagInfo(
                "disable_sliding_window", FlagType.BOOLEAN, has_negative=True
            ),
            "allow_deprecated_quantization": FlagInfo(
                "allow_deprecated_quantization", FlagType.BOOLEAN, has_negative=True
            ),
            # Value flags
            "max_model_len": FlagInfo("max_model_len", FlagType.VALUE),
            "tensor_parallel_size": FlagInfo("tensor_parallel_size", FlagType.VALUE),
            "gpu_memory_utilization": FlagInfo(
                "gpu_memory_utilization", FlagType.VALUE
            ),
            "max_num_seqs": FlagInfo("max_num_seqs", FlagType.VALUE),
            "max_num_batched_tokens": FlagInfo(
                "max_num_batched_tokens", FlagType.VALUE
            ),
            "port": FlagInfo("port", FlagType.VALUE),
            "host": FlagInfo("host", FlagType.VALUE),
            "seed": FlagInfo("seed", FlagType.VALUE),
            "block_size": FlagInfo("block_size", FlagType.VALUE),
            "max_parallel_loading_workers": FlagInfo(
                "max_parallel_loading_workers", FlagType.VALUE
            ),
            "renderer_num_workers": FlagInfo("renderer_num_workers", FlagType.VALUE),
            "shutdown_timeout": FlagInfo("shutdown_timeout", FlagType.VALUE),
            "max_logprobs": FlagInfo("max_logprobs", FlagType.VALUE),
            "max_log_len": FlagInfo("max_log_len", FlagType.VALUE),
            # Choice flags
            "dtype": FlagInfo(
                "dtype",
                FlagType.CHOICE,
                choices=["auto", "bfloat16", "float16", "float32", "half", "float"],
            ),
            "kv_cache_dtype": FlagInfo(
                "kv_cache_dtype",
                FlagType.CHOICE,
                choices=["auto", "bfloat16", "float16", "fp8", "fp8_e4m3", "fp8_e5m2"],
            ),
            "tokenizer_mode": FlagInfo(
                "tokenizer_mode",
                FlagType.CHOICE,
                choices=["auto", "mistral", "slow", "hf"],
            ),
            "load_format": FlagInfo(
                "load_format",
                FlagType.CHOICE,
                choices=["auto", "pt", "safetensors", "npcache", "dummy"],
            ),
            "quantization": FlagInfo(
                "quantization",
                FlagType.CHOICE,
                choices=[
                    "awq",
                    "gptq",
                    "squeezellm",
                    "fp8",
                    "marlin",
                    "gptq_marlin",
                    "awq_marlin",
                    "bitsandbytes",
                    "gguf",
                    "aqlm",
                    "quark",
                ],
            ),
        }

    def get_flags(self) -> Dict[str, FlagInfo]:
        """Get all vLLM flags, using cache if available."""
        if self._flag_cache is not None:
            return self._flag_cache

        # Try to load from cache file
        if self._cache_file and self._cache_file.exists():
            try:
                with open(self._cache_file, "r") as f:
                    cache_data = json.load(f)

                self._flag_cache = {
                    name: FlagInfo(
                        name=name,
                        flag_type=FlagType(info["flag_type"]),
                        has_negative=info.get("has_negative", False),
                        choices=info.get("choices"),
                    )
                    for name, info in cache_data.items()
                }
                return self._flag_cache
            except Exception:
                pass  # Fall through to parse

        # Parse from vLLM
        self._flag_cache = self._parse_help_output()
        return self._flag_cache

    def validate_and_format(
        self, config: Dict[str, Any], remove_invalid: bool = True
    ) -> ValidationResult:
        """
        Validate and format vLLM arguments from config dict.

        Args:
            config: Dict with flag names (snake_case) and values
            remove_invalid: If True, remove invalid flags. If False, raise error.

        Returns:
            ValidationResult with valid_args, removed_flags, and warnings
        """
        flags = self.get_flags()
        result = ValidationResult()

        for key, value in config.items():
            flag_name = key.lower().strip()

            # Skip special internal keys
            if flag_name.startswith("_") or flag_name in ["model", "model_id"]:
                continue

            if flag_name not in flags:
                result.removed_flags.append((key, f"Unknown flag: {key}"))
                result.warnings.append(f"Unknown flag '{key}' removed from config")
                continue

            flag_info = flags[flag_name]
            kebab_name = flag_name.replace("_", "-")

            if flag_info.flag_type == FlagType.BOOLEAN:
                # Boolean flag handling
                if isinstance(value, bool):
                    if value:
                        result.valid_args.append(f"--{kebab_name}")
                    elif flag_info.has_negative:
                        result.valid_args.append(f"--no-{kebab_name}")
                    # If False and no negative variant, omit entirely
                elif isinstance(value, str):
                    if value.lower() in ("true", "yes", "1", "on"):
                        result.valid_args.append(f"--{kebab_name}")
                    elif value.lower() in ("false", "no", "0", "off"):
                        if flag_info.has_negative:
                            result.valid_args.append(f"--no-{kebab_name}")
                    else:
                        result.removed_flags.append(
                            (key, f"Invalid boolean value: {value}")
                        )
                        result.warnings.append(
                            f"Flag '{key}' has invalid boolean value '{value}' - removed"
                        )
                else:
                    result.removed_flags.append(
                        (key, f"Non-boolean value for boolean flag: {value}")
                    )
                    result.warnings.append(
                        f"Flag '{key}' expected boolean but got {type(value).__name__} - removed"
                    )

            elif flag_info.flag_type == FlagType.CHOICE:
                # Choice flag handling
                if value is None or value == "":
                    result.removed_flags.append((key, "Empty value for choice flag"))
                    result.warnings.append(f"Flag '{key}' has empty value - removed")
                elif flag_info.choices and str(value).lower() not in [
                    c.lower() for c in flag_info.choices
                ]:
                    result.removed_flags.append((key, f"Invalid choice: {value}"))
                    result.warnings.append(
                        f"Flag '{key}' has invalid choice '{value}' - removed"
                    )
                else:
                    result.valid_args.extend([f"--{kebab_name}", str(value)])

            else:  # VALUE type
                # Value flag handling
                if value is None or value == "":
                    result.removed_flags.append((key, "Missing required value"))
                    result.warnings.append(
                        f"Flag '{key}' requires a value but none provided - removed"
                    )
                else:
                    result.valid_args.extend([f"--{kebab_name}", str(value)])

        return result

    def format_for_cli(self, config: Dict[str, Any]) -> List[str]:
        """
        Convenience method to format config for CLI, removing invalid flags.

        Args:
            config: Dict with flag names and values

        Returns:
            List of CLI arguments
        """
        result = self.validate_and_format(config, remove_invalid=True)

        if result.warnings:
            print("Flag validation warnings:", file=sys.stderr)
            for warning in result.warnings:
                print(f"  ⚠ {warning}", file=sys.stderr)

        return result.valid_args

    def get_summary(self, result: ValidationResult) -> str:
        """Get a formatted summary of validation results."""
        lines = []

        if result.valid_args:
            lines.append(
                f"Valid flags ({len(result.valid_args) // 2}): {' '.join(result.valid_args[:10])}{'...' if len(result.valid_args) > 10 else ''}"
            )

        if result.removed_flags:
            lines.append(f"\nRemoved flags ({len(result.removed_flags)}):")
            for flag, reason in result.removed_flags:
                lines.append(f"  ✗ --{flag.replace('_', '-')}: {reason}")

        return "\n".join(lines)


def validate_vllm_flags(
    config: Dict[str, Any],
    model_dir: Optional[str] = None,
    vllm_binary: Optional[str] = None,
    log_warnings: bool = True,
) -> List[str]:
    validator = VLLMFlagValidator(
        vllm_binary=vllm_binary,
        model_dir=model_dir,
    )
    return validator.format_for_cli(config)


def main():
    """CLI interface for flag validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate vLLM flags")
    parser.add_argument("--model-dir", help="Model directory (to find vLLM)")
    parser.add_argument("--vllm-binary", help="Path to vllm binary")
    parser.add_argument("--config-file", help="JSON/YAML config file to validate")
    parser.add_argument(
        "--format", choices=["cli", "json"], default="cli", help="Output format"
    )

    args = parser.parse_args()

    # Initialize validator
    validator = VLLMFlagValidator(
        vllm_binary=args.vllm_binary, model_dir=args.model_dir
    )

    # Load config if provided
    config = {}
    if args.config_file:
        import yaml

        with open(args.config_file, "r") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                # Handle nested smoke_test.vllm_args format
                if "smoke_test" in data and "vllm_args" in data["smoke_test"]:
                    config = data["smoke_test"]["vllm_args"]
                else:
                    config = data

    # Validate
    result = validator.validate_and_format(config)

    if args.format == "json":
        output = {
            "valid_args": result.valid_args,
            "removed_flags": [
                {"flag": flag, "reason": reason}
                for flag, reason in result.removed_flags
            ],
            "warnings": result.warnings,
        }
        print(json.dumps(output, indent=2))
    else:
        print(validator.get_summary(result))
        print("\nFormatted CLI args:")
        print(" ".join(result.valid_args))


if __name__ == "__main__":
    main()
