"""
Backend Selector for vLLM

Determines optimal attention backend based on:
1. Hardware capabilities (GPU type, compute capability)
2. Model architecture (some models have backend-specific issues)
3. Context length requirements
4. Quantization type

The philosophy: Start with the best backend for the hardware, fall back if it fails.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml


def get_hardware_recommended_backend(gpu_name: str, gpu_memory_gb: float) -> str:
    """
    Get recommended attention backend based on hardware.
    
    Args:
        gpu_name: GPU name (e.g., "NVIDIA H200")
        gpu_memory_gb: GPU memory in GB
        
    Returns:
        Recommended backend: FLASHINFER, FLASH_ATTN, or XFORMERS
    """
    gpu_name_upper = gpu_name.upper()
    
    # H100/H200 - FLASHINFER is best
    if "H100" in gpu_name_upper or "H200" in gpu_name_upper or gpu_memory_gb > 100:
        return "FLASHINFER"
    
    # A100 - FLASH_ATTN is stable and well-tested
    if "A100" in gpu_name_upper:
        return "FLASH_ATTN"
    
    # RTX 40 series, L4, etc. - FLASH_ATTN if available
    if any(x in gpu_name_upper for x in ["RTX 40", "L4", "L40"]):
        return "FLASH_ATTN"
    
    # Older GPUs, consumer cards - XFORMERS for compatibility
    return "XFORMERS"


def get_model_backend_preferences(model_id: str, model_config: Optional[Dict] = None) -> List[str]:
    """
    Get backend preferences for a specific model.
    
    Some models have known issues with certain backends.
    Returns ordered list of backends to try (best first).
    
    Args:
        model_id: HuggingFace model ID
        model_config: Optional model config dict
        
    Returns:
        List of backends in priority order
    """
    model_id_lower = model_id.lower()
    
    # Check for known problematic combinations
    # Format: {pattern: ([avoid_backends], [prefer_backends])}
    special_cases = {
        # MoE models - FLASHINFER has optimizations but FLASH_ATTN is more stable
        "mixtral": ([], ["FLASHINFER", "FLASH_ATTN"]),
        "deepseek-v3": ([], ["FLASHINFER", "FLASH_ATTN"]),
        "dbrx": ([], ["FLASHINFER", "FLASH_ATTN"]),
        
        # Vision models - sometimes have issues with FLASHINFER
        "llava": (["FLASHINFER"], ["FLASH_ATTN", "XFORMERS"]),
        "vision": ([], ["FLASH_ATTN", "FLASHINFER"]),
        
        # Quantized models - FLASH_ATTN often more stable
        "awq": ([], ["FLASH_ATTN", "FLASHINFER"]),
        "gptq": ([], ["FLASH_ATTN", "FLASHINFER"]),
        
        # Older architectures - XFORMERS for compatibility
        "gpt2": (["FLASHINFER"], ["FLASH_ATTN", "XFORMERS"]),
        "gpt-neo": (["FLASHINFER"], ["FLASH_ATTN", "XFORMERS"]),
        "bloom": (["FLASHINFER"], ["FLASH_ATTN", "XFORMERS"]),
        "falcon": ([], ["FLASH_ATTN", "FLASHINFER"]),
    }
    
    # Find matching special case
    avoid_backends = []
    prefer_backends = []
    
    for pattern, (avoid, prefer) in special_cases.items():
        if pattern in model_id_lower:
            avoid_backends.extend(avoid)
            prefer_backends.extend(prefer)
            break
    
    # Build ordered list
    all_backends = ["FLASHINFER", "FLASH_ATTN", "FLASH_ATTN_VLLM_V1", "XFORMERS"]
    
    # Start with preferred backends
    result = [b for b in prefer_backends if b not in avoid_backends]
    
    # Add remaining backends
    for backend in all_backends:
        if backend not in result and backend not in avoid_backends:
            result.append(backend)
    
    return result if result else all_backends


def select_optimal_backend(
    model_id: str,
    gpu_name: str,
    gpu_memory_gb: float,
    model_config: Optional[Dict] = None,
    quantization: Optional[str] = None,
    context_length: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """
    Select optimal attention backend for model + hardware combination.
    
    Args:
        model_id: HuggingFace model ID
        gpu_name: GPU name
        gpu_memory_gb: GPU memory in GB
        model_config: Optional model config
        quantization: Quantization type (awq, gptq, fp8, etc.)
        context_length: Target context length
        
    Returns:
        Tuple of (primary_backend, fallback_backends)
    """
    # Get hardware recommendation
    hardware_backend = get_hardware_recommended_backend(gpu_name, gpu_memory_gb)
    
    # Get model preferences
    model_backends = get_model_backend_preferences(model_id, model_config)
    
    # If model has specific preferences, use them
    # But ensure hardware-compatible backend is in the list
    if model_backends and model_backends[0] != hardware_backend:
        # Check if hardware-recommended backend is compatible with model
        if hardware_backend in model_backends:
            # Move hardware backend to front if model supports it
            model_backends.remove(hardware_backend)
            model_backends.insert(0, hardware_backend)
    
    # Handle quantization-specific preferences
    if quantization in ["awq", "gptq", "int4", "int8"]:
        # Quantized models often work better with FLASH_ATTN
        if "FLASH_ATTN" in model_backends and model_backends[0] == "FLASHINFER":
            # Swap FLASH_ATTN to front for quantized models
            model_backends.remove("FLASH_ATTN")
            model_backends.insert(0, "FLASH_ATTN")
    
    # Very long context (>64K) - FLASHINFER or FLASH_ATTN required
    if context_length and context_length > 65536:
        # Filter to only backends that support long context
        long_context_backends = ["FLASHINFER", "FLASH_ATTN", "FLASH_ATTN_VLLM_V1"]
        model_backends = [b for b in model_backends if b in long_context_backends]
        if not model_backends:
            model_backends = ["FLASHINFER", "FLASH_ATTN"]
    
    primary = model_backends[0] if model_backends else hardware_backend
    fallbacks = model_backends[1:] if len(model_backends) > 1 else []
    
    return primary, fallbacks


def get_backend_compatibility_notes(backend: str, model_id: str) -> Optional[str]:
    """
    Get compatibility notes for a backend + model combination.
    
    Args:
        backend: Backend name
        model_id: Model ID
        
    Returns:
        Warning message if there are known issues, None otherwise
    """
    model_id_lower = model_id.lower()
    
    notes = {
        "FLASHINFER": {
            "patterns": ["gpt2", "gpt-neo", "bloom"],
            "note": "FLASHINFER may have issues with older architectures. FLASH_ATTN recommended."
        },
        "XFORMERS": {
            "patterns": [],
            "note": "XFORMERS is slower but most compatible. Use only if other backends fail."
        },
    }
    
    for backend_name, info in notes.items():
        if backend == backend_name:
            for pattern in info["patterns"]:
                if pattern in model_id_lower:
                    return info["note"]
    
    return None


# Load from flag_knowledge_base.yaml if available
def _load_hardware_recommendations() -> Dict:
    """Load hardware recommendations from flag knowledge base."""
    kb_path = Path(__file__).parent.parent.parent / "configs" / "flag_knowledge_base.yaml"
    if kb_path.exists():
        with open(kb_path) as f:
            kb = yaml.safe_load(f)
        return kb.get("hardware_recommendations", {})
    return {}


if __name__ == "__main__":
    # Test examples
    test_cases = [
        ("meta-llama/Llama-3.1-70B", "NVIDIA H200", 141),
        ("meta-llama/Llama-3.1-8B", "NVIDIA A100", 80),
        ("deepseek-ai/DeepSeek-V3", "NVIDIA H200", 141),
        ("llava-hf/llava-1.5-7b", "NVIDIA H100", 80),
        ("gpt2", "NVIDIA RTX 4090", 24),
    ]
    
    print("Backend Selection Tests:")
    print("=" * 80)
    
    for model_id, gpu_name, gpu_mem in test_cases:
        primary, fallbacks = select_optimal_backend(model_id, gpu_name, gpu_mem)
        note = get_backend_compatibility_notes(primary, model_id)
        
        print(f"\n{model_id}")
        print(f"  Hardware: {gpu_name} ({gpu_mem}GB)")
        print(f"  Primary: {primary}")
        print(f"  Fallbacks: {fallbacks}")
        if note:
            print(f"  Note: {note}")
