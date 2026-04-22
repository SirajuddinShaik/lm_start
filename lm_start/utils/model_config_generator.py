"""Generate model-specific vLLM configuration."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class ModelConfigGenerator:
    """Generate vLLM config specific to a model architecture."""

    # Architecture-specific flag rules
    ARCHITECTURE_RULES = {
        "Gemma4ForConditionalGeneration": {
            "description": "Gemma 4 architecture with heterogeneous attention",
            "incompatible_flags": {
                "attention_backend": ["FLASH_ATTN", "FLASHINFER"],
                "flashinfer_backend": "all",
            },
            "mandatory_flags": {"trust_remote_code": True},
            "recommended_defaults": {
                "attention_backend": None,
                "kv_cache_dtype": "auto",
                "enable_chunked_prefill": True,
            },
            "constraints": {
                "head_dims": "heterogeneous (256, 512)",
                "note": "Must use auto-detect or TRITON_ATTN only",
            },
        },
        "DeepseekV3ForCausalLM": {
            "description": "DeepSeek V3 MoE architecture",
            "incompatible_flags": {},
            "mandatory_flags": {
                "trust_remote_code": True,
                "enable_expert_parallel": True,
            },
            "recommended_defaults": {"max_model_len": 65536, "tensor_parallel_size": 8},
            "constraints": {"architecture": "MoE", "params": "671B total"},
        },
        "Qwen2_5ForCausalLM": {
            "description": "Qwen 2.5 architecture",
            "incompatible_flags": {},
            "mandatory_flags": {"trust_remote_code": False},
            "recommended_defaults": {"attention_backend": "FLASH_ATTN"},
        },
        "LlamaForCausalLM": {
            "description": "Standard Llama architecture",
            "incompatible_flags": {},
            "mandatory_flags": {},
            "recommended_defaults": {"attention_backend": "FLASH_ATTN"},
        },
    }

    # Flag applicability rules based on model characteristics
    FLAG_RULES = {
        "attention_backend": {
            "depends_on": ["head_dim_compatible", "architecture"],
            "auto_detect_recommended": True,
        },
        "enable_chunked_prefill": {
            "recommended_for_long_context": True,
            "threshold": 32768,
        },
        "tensor_parallel_size": {
            "depends_on": ["model_size", "gpu_memory", "gpu_count"]
        },
        "kv_cache_dtype": {
            "fp8_requires": {"compute_capability": ">= 8.9"},
            "default": "auto",
        },
    }

    def __init__(self, model_dir: Path, vllm_flags: Dict[str, Any]):
        self.model_dir = Path(model_dir)
        self.vllm_flags = vllm_flags
        self.model_config = self._load_model_config()
        self.architecture = self.model_config.get("architectures", ["Unknown"])[0]

    def _load_model_config(self) -> Dict[str, Any]:
        """Load model config from hub_configs."""
        config_path = self.model_dir / ".llm-context" / "hub_configs" / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}

    def _get_architecture_rules(self) -> Dict[str, Any]:
        """Get rules for this architecture."""
        return self.ARCHITECTURE_RULES.get(self.architecture, {})

    def _is_flag_applicable(self, flag_name: str) -> tuple[bool, str]:
        """Check if flag is applicable to this model."""
        rules = self._get_architecture_rules()

        # Check incompatible flags
        incompatible = rules.get("incompatible_flags", {})
        if flag_name in incompatible:
            return (
                False,
                f"Incompatible with {self.architecture}: {incompatible[flag_name]}",
            )

        # Check architecture-specific constraints
        if flag_name == "attention_backend":
            if "heterogeneous" in str(rules.get("constraints", {})).lower():
                return True, "Must use auto-detect or TRITON_ATTN only"

        return True, "Applicable"

    def _get_smart_default(self, flag_name: str, flag_info: Dict) -> Any:
        """Get smart default based on model characteristics."""
        rules = self._get_architecture_rules()
        recommended = rules.get("recommended_defaults", {})

        if flag_name in recommended:
            return recommended[flag_name]

        # Use vLLM's default
        return flag_info.get("default")

    def generate_config(self) -> Dict[str, Any]:
        """Generate model-specific configuration."""
        applicable_flags = {}
        incompatible_flags = {}

        for flag_name, flag_info in self.vllm_flags.items():
            is_applicable, reason = self._is_flag_applicable(flag_name)

            if is_applicable:
                applicable_flags[flag_name] = {
                    "default": self._get_smart_default(flag_name, flag_info),
                    "recommended": self._get_smart_default(flag_name, flag_info),
                    "choices": flag_info.get("choices"),
                    "type": flag_info.get("type"),
                    "help": flag_info.get("help"),
                    "applicability_note": reason if reason != "Applicable" else None,
                }
            else:
                incompatible_flags[flag_name] = {
                    "reason": reason,
                    "vllm_default": flag_info.get("default"),
                }

        # Get mandatory flags
        rules = self._get_architecture_rules()
        mandatory = rules.get("mandatory_flags", {})

        return {
            "metadata": {
                "model_id": self.model_dir.name,
                "architecture": self.architecture,
                "generated_at": datetime.now().isoformat(),
                "vllm_version": "detected",
            },
            "applicable_flags": applicable_flags,
            "incompatible_flags": incompatible_flags,
            "mandatory_flags": mandatory,
            "architecture_notes": rules.get("constraints", {}),
            "recommendations": {
                "preferred_backend": self._get_preferred_backend(),
                "memory_optimization": self._get_memory_recommendations(),
                "performance_tips": self._get_performance_tips(),
            },
        }

    def _get_preferred_backend(self) -> str:
        """Determine preferred attention backend."""
        rules = self._get_architecture_rules()
        defaults = rules.get("recommended_defaults", {})
        backend = defaults.get("attention_backend")
        if backend is None:
            return "auto-detect (recommended for this architecture)"
        return backend

    def _get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recs = []
        rules = self._get_architecture_rules()

        if "long_context" in str(rules.get("constraints", {})).lower():
            recs.append("Enable chunked prefill for long contexts")

        recs.append("Use --gpu-memory-utilization 0.85-0.95 based on GPU memory")

        return recs

    def _get_performance_tips(self) -> List[str]:
        """Get performance optimization tips."""
        return [
            "Start with auto-detected settings",
            "Benchmark before and after changes",
            "Monitor TTFT and throughput metrics",
        ]

    def save(self, output_path: Optional[Path] = None) -> Path:
        """Generate and save configuration."""
        config = self.generate_config()

        if output_path is None:
            output_path = (
                self.model_dir
                / ".llm-context"
                / "model-context"
                / "vllm_model_config.yaml"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return output_path


def generate_model_config(model_dir: str, vllm_flags: Dict[str, Any]) -> Path:
    """Generate model-specific config file."""
    generator = ModelConfigGenerator(Path(model_dir), vllm_flags)
    return generator.save()
