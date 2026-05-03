#!/bin/bash
#
# create_venv.sh - Create virtual environment and install vLLM
#
# Usage: ./create_venv.sh <model_dir> [python_version] [vllm_version]
#
# Arguments:
#   model_dir     - Directory to create .venv in (required)
#   python_version - Python version to use (default: from model_info.json or 3.11)
#   vllm_version   - vLLM version to install (default: latest)
#
# Environment variables:
#   HF_HOME        - HuggingFace cache directory
#   CUDA_VERSION   - CUDA version for PyTorch (auto-detected if not set)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored status
status() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

CONFIG_FILE="${HOME}/.lm-start/config/system.yaml"
export CONFIG_FILE

if [ ! -f "$CONFIG_FILE" ]; then
    status "ERROR: Config file not found at: $CONFIG_FILE"
    status "Please run: lm-start init"
    exit 1
fi

load_config() {
    python3 << 'PYTHON_EOF'
import yaml
import sys
import os

config_file = os.environ.get('CONFIG_FILE', '')
if not config_file or not os.path.exists(config_file):
    print(f"ERROR: Config file not found: {config_file}", file=sys.stderr)
    sys.exit(1)

try:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Helper to safely get nested values
    def get_nested(config, *keys, default=''):
        try:
            result = config
            for key in keys:
                result = result[key]
            return str(result) if result is not None else default
        except (KeyError, TypeError):
            return default
    
    # Extract values
    base_dir = get_nested(config, 'system', 'base_dir', default='/data/models')
    cuda_home = get_nested(config, 'device', 'cuda', 'home', default='')
    cuda_version = get_nested(config, 'device', 'cuda', 'version', default='')
    hf_home = get_nested(config, 'paths', 'hf_home', default='/data/.cache/huggingface')
    ld_library_path = get_nested(config, 'environment', 'LD_LIBRARY_PATH', default='')
    
    # Print for shell to capture
    print(f"CONFIG_BASE_DIR={base_dir}")
    print(f"CONFIG_CUDA_HOME={cuda_home}")
    print(f"CONFIG_CUDA_VERSION={cuda_version}")
    print(f"CONFIG_HF_HOME={hf_home}")
    print(f"CONFIG_LD_LIBRARY_PATH={ld_library_path}")
    
    # Also export all environment variables from config
    env_vars = get_nested(config, 'environment', default={})
    if isinstance(env_vars, dict):
        for key, value in env_vars.items():
            if value is not None:
                print(f"EXPORT_ENV_{key}={value}")
    
except Exception as e:
    print(f"ERROR: Failed to parse config: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
}

# Parse config and export values
config_output=$(load_config)
eval "$config_output"

# Validate required values
if [ -z "$CONFIG_BASE_DIR" ]; then
    error "Failed to load base_dir from config: $CONFIG_FILE"
fi
if [ -z "$CONFIG_HF_HOME" ]; then
    error "Failed to load hf_home from config: $CONFIG_FILE"
fi

# Export environment variables from config
export HF_HOME="$CONFIG_HF_HOME"
if [ -n "$CONFIG_CUDA_HOME" ]; then
    export CUDA_HOME="$CONFIG_CUDA_HOME"
    export CUDA_ROOT="$CONFIG_CUDA_HOME"
fi
if [ -n "$CONFIG_LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="$CONFIG_LD_LIBRARY_PATH"
fi

# Export all environment variables from config
for env_var in $(compgen -v | grep '^EXPORT_ENV_'); do
    var_name=${env_var#EXPORT_ENV_}
    var_value=${!env_var}
    export "$var_name"="$var_value"
done

load_config

# Arguments
MODEL_DIR="${1:-.}"
PYTHON_VERSION="${2:-}"
VLLM_VERSION="${3:-latest}"
INSTALL_PYTORCH="${INSTALL_PYTORCH:-true}"  # Set to "false" to skip PyTorch install (use vLLM's bundled PyTorch)
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-}"  # PyTorch index URL for specific CUDA version (e.g., https://download.pytorch.org/whl/cu129)

# Validate model directory
if [ ! -d "$MODEL_DIR" ]; then
    error "Model directory does not exist: $MODEL_DIR"
fi

cd "$MODEL_DIR"

# Load model info if available
if [ -f "model_info.json" ]; then
    status "Loading model_info.json..."
    if command -v jq &> /dev/null; then
        [ -z "$PYTHON_VERSION" ] && PYTHON_VERSION=$(jq -r '.python_version // "3.11"' model_info.json)
        [ "$VLLM_VERSION" = "latest" ] && VLLM_VERSION=$(jq -r '.vllm_version // "latest"' model_info.json)
    else
        warn "jq not installed, using defaults"
        [ -z "$PYTHON_VERSION" ] && PYTHON_VERSION="3.11"
    fi
else
    [ -z "$PYTHON_VERSION" ] && PYTHON_VERSION="3.11"
fi

status "Python version: $PYTHON_VERSION"
status "vLLM version: $VLLM_VERSION"

# Detect CUDA version (prefer config, fallback to nvcc, NEVER use nvidia-smi)
# nvidia-smi shows DRIVER CUDA version which can be higher than toolkit
# We need the TOOLKIT version for installing wheels
detect_cuda_version() {
    # First check if config has CUDA version
    if [ -n "$CONFIG_CUDA_VERSION" ]; then
        status "Using CUDA version from config: $CONFIG_CUDA_VERSION"
        echo "$CONFIG_CUDA_VERSION"
        return
    fi
    
    # Check nvcc (most reliable - this is the actual toolkit version)
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
        if [ -n "$cuda_version" ]; then
            status "Detected CUDA toolkit version from nvcc: $cuda_version"
            echo "$cuda_version"
            return
        fi
    fi
    
    # Default to 12.9
    warn "Could not detect CUDA version, defaulting to 12.9"
    echo "12.9"
}

CUDA_VERSION="${CUDA_VERSION:-$(detect_cuda_version)}"
status "CUDA version: $CUDA_VERSION"

# Check if Python has venv AND ensurepip modules available
# Both are required to create a virtual environment
check_venv_available() {
    local python_cmd="$1"
    # Check both venv and ensurepip modules
    if ! "$python_cmd" -c "import venv" 2>/dev/null; then
        return 1
    fi
    if ! "$python_cmd" -c "import ensurepip" 2>/dev/null; then
        return 1
    fi
    return 0
}

# Check if Python version is available with smart fallback
find_best_python() {
    local requested_version="${1:-3}"
    local preferred_versions=("$requested_version" "3.12" "3.11" "3.10" "3.9")
    local found_python=""
    local found_version=""
    
    # Send status to stderr so it doesn't interfere with return value
    status "Looking for Python (requested: $requested_version)..." >&2
    
    # First pass: exact version match with venv check
    for version in "${preferred_versions[@]}"; do
        # Try pythonX.Y format
        if command -v "python${version}" &> /dev/null; then
            found_python="python${version}"
            if check_venv_available "$found_python"; then
                found_version=$("$found_python" --version 2>&1 | awk '{print $2}')
                if [[ "$found_version" == "${version}"* ]] || [[ "$version" == "3" && "$found_version" == 3.* ]]; then
                    if [[ "$version" == "$requested_version" ]]; then
                        echo "$found_python"
                        return 0
                    fi
                fi
            fi
        fi
        
        # Try python3.X format
        local short_version="${version//3./}"
        if command -v "python3.${short_version}" &> /dev/null; then
            found_python="python3.${short_version}"
            if check_venv_available "$found_python"; then
                found_version=$("$found_python" --version 2>&1 | awk '{print $2}')
                if [[ "$found_version" == "${version}"* ]]; then
                    if [[ "$version" == "$requested_version" ]]; then
                        echo "$found_python"
                        return 0
                    fi
                fi
            fi
        fi
    done
    
    # Second pass: fallback to any available version with venv
    for version in "${preferred_versions[@]}"; do
        if command -v "python${version}" &> /dev/null; then
            found_python="python${version}"
            if check_venv_available "$found_python"; then
                found_version=$("$found_python" --version 2>&1 | awk '{print $2}')
                warn "Python $requested_version not found, using Python $found_version instead" >&2
                echo "$found_python"
                return 0
            fi
        fi
        
        local short_version="${version//3./}"
        if command -v "python3.${short_version}" &> /dev/null; then
            found_python="python3.${short_version}"
            if check_venv_available "$found_python"; then
                found_version=$("$found_python" --version 2>&1 | awk '{print $2}')
                warn "Python $requested_version not found, using Python $found_version instead" >&2
                echo "$found_python"
                return 0
            fi
        fi
    done
    
    # Final fallback: any python3 with venv
    if command -v python3 &> /dev/null; then
        found_python="python3"
        if check_venv_available "$found_python"; then
            found_version=$(python3 --version 2>&1 | awk '{print $2}')
            warn "Specific Python version not found, using system python3 (version $found_version)" >&2
            echo "$found_python"
            return 0
        fi
    fi
    
    # Nothing found with venv support
    error "No suitable Python with venv support found. Please install python3-venv package."
    return 1
}

# Find suitable Python interpreter
PYTHON_CMD=$(find_best_python "$PYTHON_VERSION")

if [ -z "$PYTHON_CMD" ]; then
    error "No suitable Python found. Please install Python 3.9 or later.\nAvailable versions:\n$(ls -la /usr/bin/python* 2>/dev/null || echo 'No Python installations found in /usr/bin')"
fi

success "Using Python: $($PYTHON_CMD --version)"

# Create virtual environment
VENV_PATH="$MODEL_DIR/.venv"

if [ -d "$VENV_PATH" ]; then
    # Check if venv was successfully created before (validation passed)
    if [ -f "$VENV_PATH/bin/vllm" ] && [ -f "$MODEL_DIR/setup_state.json" ]; then
        status "Virtual environment already exists and validated, using it..."
        source "$VENV_PATH/bin/activate"
        success "Using existing virtual environment"
        exit 0
    else
        status "Removing incomplete virtual environment..."
        rm -rf "$VENV_PATH"
    fi
fi

status "Creating virtual environment..."
$PYTHON_CMD -m venv "$VENV_PATH"
success "Virtual environment created"

# Activate venv
source "$VENV_PATH/bin/activate"

# Upgrade pip
status "Upgrading pip..."
pip install --upgrade pip wheel setuptools

if [ "$INSTALL_PYTORCH" = "true" ]; then
    status "Installing PyTorch with CUDA 13.0 wheels..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
    success "PyTorch installed: $(python -c 'import torch; print(torch.__version__)')"
fi

# Set CUDA environment variables from config or system
if [ -n "$CONFIG_CUDA_HOME" ] && [ -d "$CONFIG_CUDA_HOME" ]; then
    status "Configuring CUDA from config: $CONFIG_CUDA_HOME"
    export CUDA_HOME="$CONFIG_CUDA_HOME"
    export CUDA_ROOT="$CONFIG_CUDA_HOME"
    export PATH="$CONFIG_CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CONFIG_CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    export CUDA_PATH="$CONFIG_CUDA_HOME"
elif [ -d "/usr/local/cuda-13.0" ] || [ "$CUDA_VERSION" = "13.0" ]; then
    status "Configuring DGX Spark environment..."
    export CUDA_HOME="/usr/local/cuda-13.0"
    export PATH="/usr/local/cuda-13.0/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH"
    export CUDA_PATH="/usr/local/cuda-13.0"
fi

# Verify CUDA and set library paths
status "Verifying CUDA installation..."

# First, set up library paths for PyTorch's bundled CUDA
PYTHON_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -n "$PYTHON_SITE_PACKAGES" ]; then
    # Add PyTorch's bundled NVIDIA libraries to LD_LIBRARY_PATH
    for lib_dir in "$PYTHON_SITE_PACKAGES/nvidia/"*/lib; do
        if [ -d "$lib_dir" ]; then
            export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
        fi
    done
fi

# Also add system CUDA paths
for cuda_path in /usr/local/cuda-13.0 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12.2 /usr/local/cuda-12.0 /usr/local/cuda; do
    if [ -d "$cuda_path/lib64" ]; then
        export LD_LIBRARY_PATH="$cuda_path/lib64:$LD_LIBRARY_PATH"
        export CUDA_HOME="${CUDA_HOME:-$cuda_path}"
    fi
done

# Verify CUDA availability
if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    success "CUDA is available with GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"
else
    warn "CUDA not available in PyTorch - will use CPU mode"
fi

# Install vLLM
# CRITICAL: vLLM wheel must match the CUDA TOOLKIT version (not driver version)
# Each vLLM version is compiled against specific CUDA toolkit libraries
# vLLM 0.20.0+ uses CUDA 13.0 libraries and requires CUDA 13.0 toolkit
# vLLM 0.19.1 and earlier use CUDA 12.x libraries

status "Installing vLLM $VLLM_VERSION with CUDA $CUDA_VERSION..."

# All installations use CUDA 13.0 wheels for consistency
# All installations use CUDA 13.0 (cu130) wheels
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)

if [ "$VLLM_VERSION" = "latest" ]; then
    VLLM_VERSION="0.19.0"
    status "Using vLLM 0.19.0 with CUDA 13.0 wheels"
else
    # Allow user override but warn if incompatible
    VLLM_MAJOR=$(echo "$VLLM_VERSION" | cut -d. -f1)
    VLLM_MINOR=$(echo "$VLLM_VERSION" | cut -d. -f2)
    
    if [ "$VLLM_MAJOR" -eq 0 ] && [ "$VLLM_MINOR" -lt 20 ]; then
        warn "vLLM $VLLM_VERSION is older than 0.20.0"
        warn "Consider using vLLM 0.20.0+ for best compatibility with CUDA 13.0 wheels"
    fi
fi

# Always use CUDA 13.0 (cu130) wheels for consistency
# This matches the successful Kimi-K2.5 configuration
status "Installing vLLM ${VLLM_VERSION} with CUDA 13.0 wheels..."
pip install "vllm==${VLLM_VERSION}" --extra-index-url "https://download.pytorch.org/whl/cu130"

success "vLLM installed: $(python -c 'import vllm; print(vllm.__version__)')"

# Install additional dependencies
status "Installing additional dependencies..."
pip install transformers accelerate sentencepiece

# Install optional dependencies based on model info
if [ -f "model_info.json" ]; then
    # Check for multimodal
    if command -v jq &> /dev/null; then
        is_multimodal=$(jq -r '.is_multimodal // false' model_info.json)
        if [ "$is_multimodal" = "true" ]; then
            status "Installing multimodal dependencies..."
            pip install pillow
        fi
        
        # Check for MoE
        is_moe=$(jq -r '.is_moe // false' model_info.json)
        if [ "$is_moe" = "true" ]; then
            status "Installing MoE dependencies..."
            pip install flash-attn --no-build-isolation || warn "flash-attn installation failed, continuing..."
        fi
    fi
fi

# Validate installation
status "Validating installation..."
python -c "
import sys
import torch
import vllm

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'vLLM: {vllm.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

success "Installation validated!"

# Validate vLLM CLI functionality
status "Validating vLLM CLI..."
VLLM_BIN="$VENV_PATH/bin/vllm"

if [ ! -f "$VLLM_BIN" ]; then
    error "vLLM CLI binary not found at $VLLM_BIN"
fi

if ! "$VLLM_BIN" serve --help > /dev/null 2>&1; then
    error "vLLM CLI is not working properly. Try: pip install --upgrade vllm"
fi
success "vLLM CLI working"

# Skip smoke test - it's now handled by setup_model.sh phase 4.1
# The smoke test has been migrated to the main setup flow

# Create activation script with library paths
cat > "$MODEL_DIR/activate.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Activate the model's virtual environment with CUDA library paths

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

# Activate the virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    . "$VENV_PATH/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Build LD_LIBRARY_PATH with NVIDIA CUDA libraries
# This is critical for vLLM to find CUDA runtime libraries

VENV_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
VENV_CUDA_FOUND=false

if [ -n "$VENV_SITE" ] && [ -d "$VENV_SITE/nvidia" ]; then
    # Add all NVIDIA library directories from venv (bundled with vLLM/PyTorch)
    # These take priority over system CUDA to ensure version compatibility
    for nvidia_pkg in "$VENV_SITE/nvidia/"*/lib; do
        if [ -d "$nvidia_pkg" ]; then
            export LD_LIBRARY_PATH="$nvidia_pkg:$LD_LIBRARY_PATH"
            VENV_CUDA_FOUND=true
        fi
    done 2>/dev/null
    # Also add triton backend
    [ -d "$VENV_SITE/triton/backends/nvidia/lib" ] && export LD_LIBRARY_PATH="$VENV_SITE/triton/backends/nvidia/lib:$LD_LIBRARY_PATH"
fi

# Only add system CUDA paths if no bundled CUDA found in venv
# This prevents version mismatches between bundled and system CUDA
if [ "$VENV_CUDA_FOUND" = "false" ]; then
    for cuda_path in /usr/local/cuda-13.0 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda; do
        [ -d "$cuda_path/lib64" ] && export LD_LIBRARY_PATH="$cuda_path/lib64:$LD_LIBRARY_PATH"
    done
fi

echo "Activated venv: $(basename "$SCRIPT_DIR")"
echo "Python: $(which python3)"

# Quick test
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" 2>/dev/null || true
python3 -c "import vllm; print('vLLM:', vllm.__version__)" 2>/dev/null || echo "vLLM not yet installed"
ACTIVATE_EOF
chmod +x "$MODEL_DIR/activate.sh"

success "Created activate.sh"

# Summary
echo ""
echo "============================================"
echo "Virtual Environment Setup Complete"
echo "============================================"
echo ""
echo "Location: $VENV_PATH"
echo "Activate: source $MODEL_DIR/activate.sh"
echo "         or: source $VENV_PATH/bin/activate"
echo ""
echo "vLLM command: vllm serve <model_id> [options]"
echo ""
