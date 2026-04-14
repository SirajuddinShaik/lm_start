#!/usr/bin/env python3
"""
Fetch model information from HuggingFace API.
Determines compatible vLLM/Python/PyTorch versions and special requirements.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests library required. Install with: pip install requests")
    sys.exit(1)


# Model architecture mappings
ARCHITECTURE_INFO = {
    # Llama family
    "LlamaForCausalLM": {
        "family": "llama",
        "requires_trust": False,
        "vllm_min": "0.3.0",
    },
    "LlamaModel": {"family": "llama", "requires_trust": False, "vllm_min": "0.3.0"},
    # Mistral family
    "MistralForCausalLM": {
        "family": "mistral",
        "requires_trust": False,
        "vllm_min": "0.3.0",
    },
    "MistralModel": {"family": "mistral", "requires_trust": False, "vllm_min": "0.3.0"},
    "MixtralForCausalLM": {
        "family": "mixtral",
        "is_moe": True,
        "requires_trust": False,
        "vllm_min": "0.3.0",
    },
    # Qwen family
    "Qwen2ForCausalLM": {
        "family": "qwen",
        "requires_trust": False,
        "vllm_min": "0.4.0",
    },
    "Qwen2MoeForCausalLM": {
        "family": "qwen-moe",
        "is_moe": True,
        "requires_trust": False,
        "vllm_min": "0.4.0",
    },
    "Qwen2_5_VLForConditionalGeneration": {
        "family": "qwen-vl",
        "is_multimodal": True,
        "requires_trust": True,
        "vllm_min": "0.6.0",
    },
    "Qwen3ForCausalLM": {
        "family": "qwen",
        "requires_trust": False,
        "vllm_min": "0.8.0",
    },
    # DeepSeek family
    "DeepseekV2ForCausalLM": {
        "family": "deepseek",
        "is_moe": True,
        "requires_trust": True,
        "vllm_min": "0.5.0",
    },
    "DeepseekV3ForCausalLM": {
        "family": "deepseek",
        "is_moe": True,
        "requires_trust": True,
        "vllm_min": "0.6.0",
    },
    "DeepseekForCausalLM": {
        "family": "deepseek",
        "requires_trust": True,
        "vllm_min": "0.4.0",
    },
    # Kimi family
    "KimiForCausalLM": {"family": "kimi", "requires_trust": True, "vllm_min": "0.6.0"},
    "Kimi2ForCausalLM": {"family": "kimi", "requires_trust": True, "vllm_min": "0.7.0"},
    # Phi family
    "Phi3ForCausalLM": {"family": "phi", "requires_trust": True, "vllm_min": "0.4.0"},
    "Phi3VForCausalLM": {
        "family": "phi-vl",
        "is_multimodal": True,
        "requires_trust": True,
        "vllm_min": "0.5.0",
    },
    # Gemma family
    "GemmaForCausalLM": {
        "family": "gemma",
        "requires_trust": False,
        "vllm_min": "0.4.0",
    },
    "Gemma2ForCausalLM": {
        "family": "gemma2",
        "requires_trust": False,
        "vllm_min": "0.5.0",
    },
    # Other architectures
    "FalconForCausalLM": {
        "family": "falcon",
        "requires_trust": False,
        "vllm_min": "0.3.0",
    },
    "MPTForCausalLM": {"family": "mpt", "requires_trust": False, "vllm_min": "0.3.0"},
    "BaichuanForCausalLM": {
        "family": "baichuan",
        "requires_trust": True,
        "vllm_min": "0.4.0",
    },
    "YiForCausalLM": {"family": "yi", "requires_trust": False, "vllm_min": "0.4.0"},
    "InternLMForCausalLM": {
        "family": "internlm",
        "requires_trust": False,
        "vllm_min": "0.4.0",
    },
    "InternLM2ForCausalLM": {
        "family": "internlm2",
        "requires_trust": False,
        "vllm_min": "0.4.0",
    },
    "CommandRForCausalLM": {
        "family": "cohere",
        "requires_trust": False,
        "vllm_min": "0.4.0",
    },
    "DbrxForCausalLM": {
        "family": "dbrx",
        "is_moe": True,
        "requires_trust": False,
        "vllm_min": "0.4.0",
    },
    "GroqForCausalLM": {"family": "groq", "requires_trust": True, "vllm_min": "0.6.0"},
}

# Reasoning models that need special parser
REASONING_MODELS = [
    "deepseek-reasoner",
    "deepseek-r1",
    "deepseek-prover",
    "qwen-qwq",
    "qwq",
]

# Multimodal models
MULTIMODAL_FAMILIES = [
    "qwen-vl",
    "phi-vl",
    "llava",
    "idefics",
    "paligemma",
    "pixtral",
]

# Version compatibility matrix
VLLM_PYTHON_COMPAT = {
    "0.3.0": {"python": ">=3.8,<3.12", "pytorch": ">=2.1.0"},
    "0.4.0": {"python": ">=3.8,<3.12", "pytorch": ">=2.2.0"},
    "0.5.0": {"python": ">=3.9,<3.12", "pytorch": ">=2.3.0"},
    "0.6.0": {"python": ">=3.9,<3.13", "pytorch": ">=2.4.0"},
    "0.7.0": {"python": ">=3.9,<3.13", "pytorch": ">=2.5.0"},
    "0.8.0": {"python": ">=3.9,<3.13", "pytorch": ">=2.5.0"},
    "latest": {"python": ">=3.9,<3.13", "pytorch": ">=2.5.0"},
}


@dataclass
class ModelInfo:
    model_id: str
    model_name: str
    architecture: Optional[str] = None
    family: str = "unknown"
    parameter_count: Optional[str] = None
    context_length: int = 4096
    is_moe: bool = False
    is_multimodal: bool = False
    is_reasoning: bool = False
    requires_trust_remote_code: bool = False
    requires_auth: bool = False
    license: str = "unknown"
    vllm_version: str = "latest"
    python_version: str = "3.11"
    pytorch_version: str = "2.5.0"
    special_args: list = None
    quantization: Optional[str] = None
    tensor_parallel_hint: int = 1
    max_model_len_hint: int = 4096
    local_model_path: Optional[str] = None
    download_size_gb: Optional[float] = None
    download_status: str = "not_downloaded"
    is_gguf: bool = False
    vllm_compatible: bool = True

    def __post_init__(self):
        if self.special_args is None:
            self.special_args = []


def extract_model_id(hf_url: str) -> str:
    """Extract model ID from HuggingFace URL or direct model ID."""
    # Handle direct model ID (e.g., "meta-llama/Llama-3-70b")
    if "/" in hf_url and not hf_url.startswith("http"):
        return hf_url

    # Handle URLs
    patterns = [
        r"huggingface\.co/([^/]+/[^/]+)",
        r"hf\.co/([^/]+/[^/]+)",
        r"huggingface\.co/models\?p=([^&]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, hf_url)
        if match:
            return match.group(1).rstrip("/")

    # If nothing matches, assume it's a model ID
    return hf_url


def fetch_hf_model_info(model_id: str) -> tuple:
    """Fetch model information from HuggingFace API.

    Returns:
        tuple: (model_data, requires_auth)
    """
    api_url = f"https://huggingface.co/api/models/{model_id}"

    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        return response.json(), False
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"ERROR: Model '{model_id}' not found on HuggingFace")
        elif response.status_code == 401:
            print(
                f"ERROR: Model '{model_id}' requires authentication. Set HF_TOKEN env var."
            )
            sys.exit(1)
        else:
            print(f"ERROR: HTTP {response.status_code}: {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch model info: {e}")
        sys.exit(1)


def estimate_download_size(model_data: dict) -> Optional[float]:
    """Estimate download size in GB from model files."""
    siblings = model_data.get("siblings", [])
    total_bytes = 0

    for sibling in siblings:
        filename = sibling.get("rfilename", "")
        if filename.endswith((".safetensors", ".bin", ".gguf", ".pt", ".pth")):
            total_bytes += sibling.get("size", 0)

    if total_bytes > 0:
        return round(total_bytes / (1024**3), 2)
    return None


def check_requires_auth(model_data: dict) -> bool:
    """Check if model requires authentication."""
    private = model_data.get("private", False)
    gated = model_data.get("gated", False)
    return private or gated


def is_gguf_model(model_id: str, model_data: dict) -> bool:
    """Check if model is a GGUF quantized model (not compatible with vLLM)."""
    model_name_lower = model_id.lower()
    if "gguf" in model_name_lower:
        return True
    
    siblings = model_data.get("siblings", [])
    for sibling in siblings:
        filename = sibling.get("rfilename", "").lower()
        if filename.endswith(".gguf"):
            return True
    
    tags = model_data.get("tags", [])
    if "gguf" in [t.lower() for t in tags]:
        return True
    
    return False


def estimate_parameters(model_id: str, model_data: dict) -> Optional[str]:
    """Estimate parameter count from model name or siblings."""
    # Check model name for parameter hints
    param_patterns = [
        (r"[-_]?(\d+)[bB](?:[-_]|$)", lambda m: f"{m.group(1)}B"),
        (r"[-_]?(\d+)[mM](?:[-_]|$)", lambda m: f"{m.group(1)}M"),
        (
            r"[-_](\d+(?:\.\d+)?)[xX](\d+)[bB]",
            lambda m: f"{m.group(1)}x{m.group(2)}B (MoE)",
        ),
    ]

    for pattern, formatter in param_patterns:
        match = re.search(pattern, model_id)
        if match:
            return formatter(match)

    # Check siblings for GGUF/binned models that might have size info
    siblings = model_data.get("siblings", [])
    for sibling in siblings:
        filename = sibling.get("rfilename", "")
        if "quantize" in filename.lower() or filename.endswith(".gguf"):
            # Extract size from filename
            size_match = re.search(r"(\d+)[bB]", filename)
            if size_match:
                return f"{size_match.group(1)}B"

    return None


def detect_architecture(model_data: dict) -> tuple:
    """Detect model architecture from config."""
    config = model_data.get("config", {})

    # Check for architectures field
    architectures = config.get("architectures", [])
    if architectures:
        arch = architectures[0]
        if arch in ARCHITECTURE_INFO:
            info = ARCHITECTURE_INFO[arch]
            return arch, info

    # Try to infer from model type
    model_type = config.get("model_type", "").lower()

    type_mapping = {
        "llama": ("LlamaForCausalLM", {"family": "llama"}),
        "mistral": ("MistralForCausalLM", {"family": "mistral"}),
        "mixtral": ("MixtralForCausalLM", {"family": "mixtral", "is_moe": True}),
        "qwen2": ("Qwen2ForCausalLM", {"family": "qwen"}),
        "qwen2_moe": ("Qwen2MoeForCausalLM", {"family": "qwen-moe", "is_moe": True}),
        "qwen2_5_vl": (
            "Qwen2_5_VLForConditionalGeneration",
            {"family": "qwen-vl", "is_multimodal": True},
        ),
        "deepseek_v2": (
            "DeepseekV2ForCausalLM",
            {"family": "deepseek", "is_moe": True},
        ),
        "deepseek_v3": (
            "DeepseekV3ForCausalLM",
            {"family": "deepseek", "is_moe": True},
        ),
        "phi3": ("Phi3ForCausalLM", {"family": "phi"}),
        "gemma": ("GemmaForCausalLM", {"family": "gemma"}),
        "gemma2": ("Gemma2ForCausalLM", {"family": "gemma2"}),
    }

    if model_type in type_mapping:
        return type_mapping[model_type]

    return None, {"family": "unknown"}


def get_context_length(model_data: dict) -> int:
    """Get context length from model config."""
    config = model_data.get("config", {})

    # Try various config keys
    keys = [
        "max_position_embeddings",
        "max_sequence_length",
        "model_max_length",
        "seq_length",
        "n_positions",
    ]

    for key in keys:
        if key in config:
            return config[key]

    # Default context lengths by family
    return 4096


def detect_special_features(model_id: str, model_data: dict, arch_info: dict) -> dict:
    """Detect special features like reasoning, multimodal, etc."""
    model_id_lower = model_id.lower()

    features = {
        "is_reasoning": False,
        "is_multimodal": False,
        "requires_trust_remote_code": arch_info.get("requires_trust", False),
    }

    # Check for reasoning models
    for pattern in REASONING_MODELS:
        if pattern in model_id_lower:
            features["is_reasoning"] = True
            break

    # Check for multimodal
    if arch_info.get("is_multimodal", False):
        features["is_multimodal"] = True
    else:
        for family in MULTIMODAL_FAMILIES:
            if family in model_id_lower or family == arch_info.get("family"):
                features["is_multimodal"] = True
                break

    # Check config for multimodal indicators
    config = model_data.get("config", {})
    if config.get("vision_config") or config.get("image_token_id"):
        features["is_multimodal"] = True

    # Check for trust_remote_code requirement in model card
    model_card = model_data.get("cardData", {})
    if model_card.get("library_name") == "transformers" and "custom_code" in str(
        model_card
    ):
        features["requires_trust_remote_code"] = True

    # Check config for trust_remote_code
    if config.get("trust_remote_code", False):
        features["requires_trust_remote_code"] = True

    return features


def get_vllm_args(model_info: ModelInfo, model_data: dict) -> list:
    """Generate vLLM arguments based on model characteristics."""
    args = []
    config = model_data.get("config", {})

    # Trust remote code
    if model_info.requires_trust_remote_code:
        args.extend(["--trust-remote-code"])

    # Reasoning parser
    if model_info.is_reasoning:
        # Determine parser type based on model
        if "deepseek" in model_info.model_id.lower():
            args.extend(["--reasoning-parser", "deepseek"])
        elif "qwq" in model_info.model_id.lower():
            args.extend(["--reasoning-parser", "deepseek"])  # QwQ uses similar format

    # MoE specific
    if model_info.is_moe:
        # MoE models often need specific settings
        args.extend(["--enable-expert-parallel"])

    # Multimodal
    if model_info.is_multimodal:
        args.extend(["--limit-mm-per-prompt", "image=5"])

    # Tensor parallel size hint based on parameter count
    if model_info.parameter_count:
        param_match = re.search(r"(\d+)", model_info.parameter_count)
        if param_match:
            params = int(param_match.group(1))
            if params >= 70:  # 70B+
                model_info.tensor_parallel_hint = 8
            elif params >= 30:  # 30-70B
                model_info.tensor_parallel_hint = 4
            elif params >= 10:  # 10-30B
                model_info.tensor_parallel_hint = 2

    # Max model len hint based on context length and VRAM estimate
    context = model_info.context_length
    if context >= 128000:
        model_info.max_model_len_hint = 32768  # Start conservative
    elif context >= 32000:
        model_info.max_model_len_hint = 16384
    else:
        model_info.max_model_len_hint = context

    # Check for chat template
    if model_info.is_reasoning or "instruct" in model_info.model_id.lower():
        args.extend(["--chat-template", "auto"])

    return args


def determine_versions(arch_info: dict, model_id: str) -> dict:
    """Determine compatible vLLM, Python, and PyTorch versions."""
    vllm_min = arch_info.get("vllm_min", "0.4.0")

    # Get version compatibility info
    compat = VLLM_PYTHON_COMPAT.get(vllm_min, VLLM_PYTHON_COMPAT["latest"])

    # Extract Python version constraint
    python_spec = compat["python"]
    # Default to 3.11 as it's well-supported
    python_version = "3.11"
    if "<3.12" in python_spec and ">=3.9" in python_spec:
        python_version = "3.11"
    elif "<3.11" in python_spec:
        python_version = "3.10"

    # Extract PyTorch version
    pytorch_spec = compat["pytorch"]
    pytorch_version = "2.5.0"  # Default to recent stable
    if ">=2.5" in pytorch_spec:
        pytorch_version = "2.5.0"
    elif ">=2.4" in pytorch_spec:
        pytorch_version = "2.4.0"
    elif ">=2.3" in pytorch_spec:
        pytorch_version = "2.3.0"

    # Use latest vLLM for now (can be pinned later)
    vllm_version = "latest"

    return {
        "vllm": vllm_version,
        "python": python_version,
        "pytorch": pytorch_version,
    }


def fetch_model_info(
    hf_url: str, output_file: Optional[str] = None, hf_home: Optional[str] = None
) -> ModelInfo:
    """Main function to fetch and process model information."""
    model_id = extract_model_id(hf_url)
    print(f"Fetching info for: {model_id}")

    model_data, requires_auth = fetch_hf_model_info(model_id)

    model_name = model_id.split("/")[-1]
    
    is_gguf = is_gguf_model(model_id, model_data)
    if is_gguf:
        print(f"WARNING: This is a GGUF model - NOT compatible with vLLM!")
        print(f"         GGUF models require llama.cpp, not vLLM.")
        print(f"         Setup will be skipped.")

    architecture, arch_info = detect_architecture(model_data)
    if architecture is None:
        print(f"WARNING: Could not detect architecture, using defaults")
        architecture = "Unknown"

    print(f"Architecture: {architecture}")
    print(f"Family: {arch_info.get('family', 'unknown')}")

    param_count = estimate_parameters(model_id, model_data)
    if param_count:
        print(f"Estimated parameters: {param_count}")

    context_length = get_context_length(model_data)
    print(f"Context length: {context_length}")

    license_info = model_data.get("license", "unknown")
    if not license_info:
        license_info = model_data.get("cardData", {}).get("license", "unknown")

    features = detect_special_features(model_id, model_data, arch_info)

    versions = determine_versions(arch_info, model_id)

    download_size = estimate_download_size(model_data)
    if download_size:
        print(f"Estimated download size: {download_size} GB")

    local_path = None
    if hf_home:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from download_model import get_model_cache_path
            local_path = get_model_cache_path(model_id, hf_home)
        except ImportError:
            local_path = f"{hf_home}/{model_id}"

    model_info = ModelInfo(
        model_id=model_id,
        model_name=model_name,
        architecture=architecture,
        family=arch_info.get("family", "unknown"),
        parameter_count=param_count,
        context_length=context_length,
        is_moe=arch_info.get("is_moe", False),
        is_multimodal=features["is_multimodal"],
        is_reasoning=features["is_reasoning"],
        requires_trust_remote_code=features["requires_trust_remote_code"],
        requires_auth=requires_auth,
        license=license_info,
        vllm_version=versions["vllm"],
        python_version=versions["python"],
        pytorch_version=versions["pytorch"],
        local_model_path=local_path,
        download_size_gb=download_size,
        download_status="not_downloaded",
        is_gguf=is_gguf,
        vllm_compatible=not is_gguf,
    )

    model_info.special_args = get_vllm_args(model_info, model_data)

    print(f"\nModel Summary:")
    print(f"  ID: {model_info.model_id}")
    print(f"  Architecture: {model_info.architecture}")
    print(f"  Family: {model_info.family}")
    print(f"  Parameters: {model_info.parameter_count or 'unknown'}")
    print(f"  Context: {model_info.context_length}")
    print(f"  MoE: {model_info.is_moe}")
    print(f"  Multimodal: {model_info.is_multimodal}")
    print(f"  Reasoning: {model_info.is_reasoning}")
    print(f"  Trust remote code: {model_info.requires_trust_remote_code}")
    print(f"  Requires auth: {model_info.requires_auth}")
    print(f"  Download size: {model_info.download_size_gb or 'unknown'} GB")
    print(f"  vLLM compatible: {model_info.vllm_compatible}")
    if model_info.is_gguf:
        print(f"  *** GGUF MODEL - Use llama.cpp instead of vLLM ***")
    print(f"  vLLM version: {model_info.vllm_version}")
    print(f"  Python version: {model_info.python_version}")
    print(f"  PyTorch version: {model_info.pytorch_version}")

    if model_info.special_args:
        print(f"  Special args: {' '.join(model_info.special_args)}")

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        info_dict = asdict(model_info)

        with open(output_path, "w") as f:
            json.dump(info_dict, f, indent=2)
        print(f"\nModel info saved to: {output_file}")

    return model_info


def main():
    parser = argparse.ArgumentParser(
        description="Fetch model information from HuggingFace"
    )
    parser.add_argument(
        "model", help="HuggingFace model URL or ID (e.g., meta-llama/Llama-3-70b)"
    )
    parser.add_argument(
        "-o", "--output", help="Output JSON file path (default: stdout)"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON to stdout")
    parser.add_argument("--hf-home", help="HuggingFace cache directory for local path")

    args = parser.parse_args()

    model_info = fetch_model_info(args.model, args.output, args.hf_home)

    if args.json and not args.output:
        print(json.dumps(asdict(model_info), indent=2))


if __name__ == "__main__":
    main()
