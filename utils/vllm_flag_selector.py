"""
VLLM Flag Selector - Intelligent Configuration Selection

Uses dynamically extracted vLLM flags to:
1. Select optimal flags based on optimization profile
2. Suggest OOM mitigation strategies
3. Reason about flag interactions and dependencies
4. Systematically explore the configuration space

This is the "master of inference" intelligence layer.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class OptimizationProfile(str, Enum):
    QUALITY = "quality"
    BALANCED = "balanced"
    SPEED = "speed"


@dataclass
class FlagRecommendation:
    """A recommendation for a specific flag value."""
    flag_name: str
    value: Any
    reason: str
    confidence: float = 1.0  # 0.0 to 1.0
    expected_impact: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "flag": self.flag_name,
            "value": self.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "expected_impact": self.expected_impact,
        }


@dataclass
class OOMMitigationStrategy:
    """A strategy for mitigating OOM errors."""
    priority: int  # Lower = higher priority
    flag_name: str
    current_value: Any
    suggested_value: Any
    reason: str
    expected_memory_reduction: str  # e.g., "20-30%"
    trade_off: str  # What you lose by applying this
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority,
            "flag": self.flag_name,
            "from": self.current_value,
            "to": self.suggested_value,
            "reason": self.reason,
            "memory_reduction": self.expected_memory_reduction,
            "trade_off": self.trade_off,
        }


class VLLMFlagSelector:
    """
    Intelligent selector for vLLM configuration flags.
    
    Uses dynamically extracted flag definitions from the model's venv
    to make informed configuration decisions.
    """
    
    # Hardware capabilities
    FP8_CAPABLE_GPUS = ["H100", "H200", "H800", "MI300X"]
    
    def __init__(self, model_dir: Path):
        """
        Initialize with model directory.
        
        Expects vllm_available_flags.json to exist (run extract_vllm_flags.py first)
        """
        self.model_dir = Path(model_dir)
        self.flags: Dict[str, Any] = {}
        self.categories: Dict[str, List[str]] = {}
        self.oom_flags: Dict[str, str] = {}
        self.profile_recommendations: Dict[str, Dict[str, Any]] = {}
        
        self._load_flags()
    
    def _load_flags(self):
        """Load dynamically extracted flags."""
        flags_path = self.model_dir / "vllm_available_flags.json"
        
        if not flags_path.exists():
            raise FileNotFoundError(
                f"vllm_available_flags.json not found. "
                f"Run: /model/.venv/bin/python extract_vllm_flags.py {self.model_dir}"
            )
        
        with open(flags_path) as f:
            data = json.load(f)
        
        self.flags = data.get("flags", {})
        self.categories = data.get("categories", {})
        self.oom_flags = data.get("oom_relevant_flags", {})
        self.profile_recommendations = data.get("profile_recommendations", {})
        
        # Add some flags that might not be in oom_relevant but are useful
        self._enrich_oom_flags()
    
    def _enrich_oom_flags(self):
        """Add additional OOM-relevant flags with strategies."""
        additional_oom = {
            "attention_backend": "Try FLASH_ATTN or XFORMERS if FLASHINFER OOMs",
            "enable_prefix_caching": "Enable to reduce memory for repeated prompts",
            "max_cudagraph_capture_size": "Reduce to lower CUDA graph memory",
        }
        self.oom_flags.update(additional_oom)
    
    def get_applicable_flags(
        self,
        gpu_type: Optional[str] = None,
        is_moe: bool = False,
        optimization_profile: OptimizationProfile = OptimizationProfile.BALANCED,
    ) -> List[FlagRecommendation]:
        """
        Get list of applicable flag recommendations for the model.
        
        This is the main entry point for the agent to discover what flags
        it can/should use for the current model.
        """
        recommendations = []
        
        # Get profile-specific base recommendations
        profile_recs = self.profile_recommendations.get(optimization_profile.value, {})
        
        for flag_name, rec in profile_recs.items():
            if flag_name not in self.flags:
                continue
            
            flag_info = self.flags[flag_name]
            
            # Validate the recommendation is valid
            choices = flag_info.get("choices")
            value = rec.get("value")
            
            if choices and value not in choices:
                # Skip if recommended value not in choices (e.g., fp8 not available)
                continue
            
            # Check hardware requirements
            if flag_name == "kv_cache_dtype" and value in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
                if not gpu_type or not any(g in gpu_type.upper() for g in self.FP8_CAPABLE_GPUS):
                    continue  # Skip FP8 on non-FP8 hardware
            
            recommendations.append(FlagRecommendation(
                flag_name=flag_name,
                value=value,
                reason=rec.get("reason", ""),
                confidence=0.9,
                expected_impact=self._get_expected_impact(flag_name, value),
            ))
        
        # Add MoE-specific recommendations
        if is_moe:
            moe_recs = self._get_moe_recommendations()
            recommendations.extend(moe_recs)
        
        # Sort by importance
        recommendations.sort(key=lambda r: self._get_flag_priority(r.flag_name), reverse=True)
        
        return recommendations
    
    def _get_moe_recommendations(self) -> List[FlagRecommendation]:
        """Get recommendations specific to MoE models."""
        recs = []
        
        if "enable_expert_parallel" in self.flags:
            recs.append(FlagRecommendation(
                flag_name="enable_expert_parallel",
                value=True,
                reason="REQUIRED for MoE models to distribute experts across GPUs",
                confidence=1.0,
                expected_impact="Better expert utilization, may increase memory slightly",
            ))
        
        if "all2all_backend" in self.flags:
            recs.append(FlagRecommendation(
                flag_name="all2all_backend",
                value="nccl",
                reason="NCCL is most reliable for MoE all-to-all communication",
                confidence=0.8,
                expected_impact="Reliable communication, good performance",
            ))
        
        return recs
    
    def _get_expected_impact(self, flag_name: str, value: Any) -> str:
        """Get expected impact description for a flag setting."""
        impacts = {
            ("kv_cache_dtype", "fp8"): "50% KV cache memory reduction, slight quality trade-off",
            ("kv_cache_dtype", "fp8_e4m3"): "50% KV cache memory reduction, standard FP8 format",
            ("enforce_eager", False): "CUDA graphs for better performance",
            ("enforce_eager", True): "Eager mode for debugging, slightly slower",
            ("enable_chunked_prefill", True): "Better throughput, interleaved prefill/decode",
            ("block_size", 16): "Standard block size, good for most cases",
            ("block_size", 32): "Larger blocks, less fragmentation, more memory",
            ("dtype", "bfloat16"): "Best numerical stability on Ampere+",
            ("dtype", "float16"): "Faster on older GPUs, slight precision loss",
            ("dtype", "auto"): "Let vLLM decide based on hardware",
        }
        return impacts.get((flag_name, value), f"Configure {flag_name}={value}")
    
    def _get_flag_priority(self, flag_name: str) -> int:
        """Get priority score for a flag (higher = more important)."""
        priorities = {
            "kv_cache_dtype": 100,  # High impact on memory
            "enable_chunked_prefill": 90,
            "dtype": 80,
            "enforce_eager": 70,
            "block_size": 60,
            "enable_expert_parallel": 95,  # Critical for MoE
        }
        return priorities.get(flag_name, 50)
    
    def get_oom_mitigation_strategies(
        self,
        current_config: Dict[str, Any],
        error_message: str = "",
        oom_iteration: int = 1,  # Which OOM we're handling (1st, 2nd, etc.)
    ) -> List[OOMMitigationStrategy]:
        """
        Get ordered list of OOM mitigation strategies.
        
        This is the key intelligence for the agent to handle OOM errors.
        Strategies are ordered by priority and expected effectiveness.
        """
        strategies = []
        
        # Priority 1: Reduce batch tokens (most effective)
        current_tokens = current_config.get("max_num_batched_tokens", 65536)
        if current_tokens > 4096 and oom_iteration <= 3:
            reduction_levels = [32768, 16384, 8192, 4096]
            for i, new_value in enumerate(reduction_levels):
                if new_value < current_tokens:
                    strategies.append(OOMMitigationStrategy(
                        priority=1 + i,
                        flag_name="max_num_batched_tokens",
                        current_value=current_tokens,
                        suggested_value=new_value,
                        reason=f"Reduce batch tokens from {current_tokens} to {new_value}",
                        expected_memory_reduction="15-25% per reduction",
                        trade_off="Lower batch throughput",
                    ))
        
        # Priority 2: Reduce max sequences
        current_seqs = current_config.get("max_num_seqs", 256)
        if current_seqs > 32 and oom_iteration <= 3:
            for i, new_value in enumerate([128, 64, 32]):
                if new_value < current_seqs:
                    strategies.append(OOMMitigationStrategy(
                        priority=10 + i,
                        flag_name="max_num_seqs",
                        current_value=current_seqs,
                        suggested_value=new_value,
                        reason=f"Reduce concurrent sequences from {current_seqs} to {new_value}",
                        expected_memory_reduction="10-20% per reduction",
                        trade_off="Lower concurrency",
                    ))
        
        # Priority 3: Enable chunked prefill if not enabled
        if not current_config.get("enable_chunked_prefill", True) and oom_iteration <= 2:
            strategies.append(OOMMitigationStrategy(
                priority=20,
                flag_name="enable_chunked_prefill",
                current_value=False,
                suggested_value=True,
                reason="Enable chunked prefill for better memory efficiency",
                expected_memory_reduction="10-30% for long contexts",
                trade_off="Slightly higher TTFT",
            ))
        
        # Priority 4: Reduce GPU memory utilization
        current_util = current_config.get("gpu_memory_utilization", 0.90)
        if current_util > 0.80 and oom_iteration <= 3:
            for i, new_value in enumerate([0.85, 0.80]):
                if new_value < current_util:
                    strategies.append(OOMMitigationStrategy(
                        priority=30 + i,
                        flag_name="gpu_memory_utilization",
                        current_value=current_util,
                        suggested_value=new_value,
                        reason=f"Reduce memory utilization from {current_util} to {new_value}",
                        expected_memory_reduction=f"{int((current_util - new_value) * 100)}% of GPU memory",
                        trade_off="Less KV cache capacity",
                    ))
        
        # Priority 5: KV cache dtype (FP8) - only if on capable hardware
        if "kv_cache_dtype" in self.flags and oom_iteration <= 2:
            current_kv = current_config.get("kv_cache_dtype", "auto")
            if current_kv not in ["fp8", "fp8_e4m3"]:
                strategies.append(OOMMitigationStrategy(
                    priority=40,
                    flag_name="kv_cache_dtype",
                    current_value=current_kv,
                    suggested_value="fp8",
                    reason="Use FP8 for KV cache (requires H100/H200)",
                    expected_memory_reduction="50% of KV cache memory",
                    trade_off="Slight quality reduction (usually negligible)",
                ))
        
        # Priority 6: Adjust block size
        current_block = current_config.get("block_size", 16)
        if current_block < 32 and oom_iteration <= 2:
            strategies.append(OOMMitigationStrategy(
                priority=50,
                flag_name="block_size",
                current_value=current_block,
                suggested_value=32,
                reason="Increase block size to reduce fragmentation",
                expected_memory_reduction="5-15% from reduced fragmentation",
                trade_off="Slightly less flexible allocation",
            ))
        
        # Priority 7: Try different attention backend
        current_backend = current_config.get("attention_backend", "FLASHINFER")
        if current_backend == "FLASHINFER" and oom_iteration <= 2:
            strategies.append(OOMMitigationStrategy(
                priority=60,
                flag_name="attention_backend",
                current_value="FLASHINFER",
                suggested_value="FLASH_ATTN",
                reason="Try FLASH_ATTN backend (may use less memory)",
                expected_memory_reduction="Variable, 5-10% typically",
                trade_off="Slightly lower performance",
            ))
        
        # Priority 8: Disable CUDA graphs (last resort for memory)
        if not current_config.get("enforce_eager", False) and oom_iteration >= 3:
            strategies.append(OOMMitigationStrategy(
                priority=70,
                flag_name="enforce_eager",
                current_value=False,
                suggested_value=True,
                reason="Disable CUDA graphs to free graph memory",
                expected_memory_reduction="5-10% graph overhead",
                trade_off="Significant performance reduction",
            ))
        
        # Sort by priority
        strategies.sort(key=lambda s: s.priority)
        
        return strategies
    
    def suggest_next_config(
        self,
        current_config: Dict[str, Any],
        error_type: str,
        error_message: str = "",
        attempt_number: int = 1,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Suggest next configuration based on error.
        
        Returns:
            Tuple of (new_config, reasoning) or (None, "exhausted")
        """
        if error_type == "cuda_oom":
            strategies = self.get_oom_mitigation_strategies(
                current_config,
                error_message,
                oom_iteration=attempt_number,
            )
            
            if not strategies:
                return None, "Exhausted all OOM mitigation strategies"
            
            # Get the highest priority strategy
            strategy = strategies[0]
            
            # Apply the strategy
            new_config = current_config.copy()
            new_config[strategy.flag_name] = strategy.suggested_value
            
            reasoning = (
                f"OOM mitigation (attempt {attempt_number}): "
                f"{strategy.reason}. Expected: {strategy.expected_memory_reduction}. "
                f"Trade-off: {strategy.trade_off}"
            )
            
            return new_config, reasoning
        
        elif error_type == "context_too_large":
            # Reduce context
            current_context = current_config.get("max_model_len", 32768)
            new_context = current_context // 2
            
            if new_context < 4096:
                return None, "Context reduced below minimum (4096)"
            
            new_config = current_config.copy()
            new_config["max_model_len"] = new_context
            
            return new_config, f"Context too large, reducing from {current_context} to {new_context}"
        
        elif error_type == "incompatible_format":
            # Try safetensors
            new_config = current_config.copy()
            new_config["load_format"] = "safetensors"
            return new_config, "Trying safetensors load format"
        
        return None, f"No strategy for error type: {error_type}"
    
    def get_flag_info(self, flag_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a specific flag."""
        if flag_name not in self.flags:
            return None
        
        flag_info = self.flags[flag_name].copy()
        
        # Add category info
        flag_info["categories"] = [
            cat for cat, flags in self.categories.items()
            if flag_name in flags
        ]
        
        # Add OOM relevance
        flag_info["oom_relevant"] = flag_name in self.oom_flags
        if flag_info["oom_relevant"]:
            flag_info["oom_strategy"] = self.oom_flags.get(flag_name)
        
        return flag_info
    
    def list_flags_by_category(self, category: str) -> List[str]:
        """List all flags in a category."""
        return self.categories.get(category, [])
    
    def print_summary(self):
        """Print a summary of available flags."""
        print(f"\n{'='*60}")
        print(f"VLLM Flag Selector Summary")
        print(f"{'='*60}")
        print(f"Total flags: {len(self.flags)}")
        print(f"\nBy category:")
        for cat, flags in sorted(self.categories.items()):
            if flags:
                print(f"  {cat}: {len(flags)} flags")
        print(f"\nOOM-relevant flags: {len(self.oom_flags)}")
        print(f"Profile recommendations: {len(self.profile_recommendations)}")
        print(f"{'='*60}\n")


# Convenience functions
def get_flag_selector(model_dir: Path) -> VLLMFlagSelector:
    """Create a flag selector for a model directory."""
    return VLLMFlagSelector(model_dir)


def suggest_oom_fix(
    model_dir: Path,
    current_config: Dict[str, Any],
    error_message: str = "",
    attempt: int = 1,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Quick function to get OOM fix suggestion."""
    selector = VLLMFlagSelector(model_dir)
    return selector.suggest_next_config(
        current_config,
        "cuda_oom",
        error_message,
        attempt,
    )
