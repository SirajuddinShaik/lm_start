#!/bin/bash
#
# setup_model.sh - Auto Model Deployer for vLLM + PM2
#
# Orchestrates the complete setup of a vLLM serving environment for any HuggingFace model.
# Supports state management, resumption, and tmux for long-running operations.
#
# Usage: ./setup_model.sh <hf_url_or_id> [options]
#
# Examples:
#   ./setup_model.sh Qwen/Qwen2.5-32B
#   ./setup_model.sh meta-llama/Llama-3.1-70B --optimize
#   ./setup_model.sh Qwen/Qwen2.5-32B --resume
#
# Options:
#   --base-dir DIR      Base directory for models (default: /models)
#   --port PORT         Server port (default: 8000)
#   --python VER        Python version (default: auto-detect from model)
#   --vllm VER          vLLM version (default: latest)
#   --no-venv           Skip virtual environment creation
#   --no-install        Skip dependency installation
#   --no-download       Skip model download (use existing)
#   --tmux-download     Run model download in tmux session
#   --research          Research optimal vLLM config (uses opencode)
#   --resume            Resume from interrupted setup
#   --status            Show current setup status
#   --dry-run           Show what would be done without executing
#

set -e

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use lm_start's venv for script dependencies
LMSTART_VENV="$SCRIPT_DIR/.venv"
if [ -f "$LMSTART_VENV/bin/activate" ]; then
    source "$LMSTART_VENV/bin/activate"
    # Override python/pip to use venv
    PYTHON_CMD="$LMSTART_VENV/bin/python"
    PIP_CMD="$LMSTART_VENV/bin/pip"
else
    echo "[WARN] lm_start venv not found at $LMSTART_VENV, using system python"
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

# Load base_dir from config/system.yaml using system python3
CONFIG_FILE="$SCRIPT_DIR/config/system.yaml"
if [ -f "$CONFIG_FILE" ]; then
    DEFAULT_BASE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['system']['base_dir'])" 2>/dev/null)
    if [ -z "$DEFAULT_BASE_DIR" ]; then
        error "Failed to load base_dir from config: $CONFIG_FILE"
    fi
else
    error "Config file not found: $CONFIG_FILE"
fi

DEFAULT_PORT=8000
DEVICE_CONFIG="$SCRIPT_DIR/configs/device_template.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ==============================================================================
# Helper Functions
# ==============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           Auto Model Deployer for vLLM + PM2                 ║"
    echo "║                                                              ║"
    echo "║  Setup any HuggingFace model for production serving         ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_step() {
    echo -e "\n${BOLD}${BLUE}[PHASE $1]${NC} ${BOLD}$2${NC}"
    echo "-------------------------------------------"
}

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command not found: $1"
        case "$1" in
            python3) log_info "Install Python: https://www.python.org/downloads/" ;;
            pip) log_info "Install pip: python3 -m ensurepip" ;;
            pm2) log_info "Install PM2: npm install -g pm2" ;;
            jq) log_info "Install jq: apt install jq or brew install jq" ;;
        esac
        exit 1
    fi
}

show_usage() {
    cat << EOF
Usage: $(basename "$0") <hf_url_or_id> [options]

Arguments:
  hf_url_or_id         HuggingFace model URL or ID
                       Examples:
                         Qwen/Qwen2.5-32B
                         https://huggingface.co/mistralai/Mistral-7B-v0.3

Options:
  --base-dir DIR       Base directory for models (default: $DEFAULT_BASE_DIR)
  --port PORT          Server port (default: $DEFAULT_PORT)
  --python VER         Python version (default: auto-detect)
  --vllm VER           vLLM version (default: latest)
  --tensor-parallel N  Tensor parallel size (default: auto-detect)
  --max-model-len N    Max context length (default: auto-detect)
  --no-venv            Skip virtual environment creation
  --no-install         Skip dependency installation
  --no-download        Skip model download (use existing)
  --tmux-download      Run model download in tmux session
  --optimize           Auto-optimize vLLM config (default: enabled)
  --agentic            Use OpenCode-based agentic optimization (default: enabled)
  --deep               Enable deep optimization mode (thorough backend/CUDA graph research)
  --force              Force re-optimization even if already optimized
  --deploy [COMMAND]    Deploy with PM2 after setup (commands: start, stop, restart; default: start)
  --mode PROFILE        Configuration profile (profiles: speed, balanced, quality; default: balanced)
  --resume             Resume from interrupted setup
  --status             Show current setup status
  --dry-run            Show what would be done without executing
  --help, -h           Show this help message

Examples:
  # Basic setup
  $(basename "$0") Qwen/Qwen2.5-32B

  # With research optimization
  $(basename "$0") meta-llama/Llama-3.1-70B --research

  # Resume interrupted setup
  $(basename "$0") Qwen/Qwen2.5-32B --resume

  # Run download in tmux (for large models)
  $(basename "$0") Qwen/Qwen2.5-32B --tmux-download

  # Deploy after setup (default: start with balanced profile)
  $(basename "$0") Qwen/Qwen2.5-32B --deploy

  # Deploy with specific profile
  $(basename "$0") Qwen/Qwen2.5-32B --deploy --mode speed

  # Deploy with specific command
  $(basename "$0") Qwen/Qwen2.5-32B --deploy stop
  $(basename "$0") Qwen/Qwen2.5-32B --deploy restart --mode quality

EOF
}

# ==============================================================================
# State Management Functions
# ==============================================================================

state_init() {
    python3 "$SCRIPT_DIR/state_manager.py" "$MODEL_DIR" --init "$MODEL_ID"
}

state_start_phase() {
    python3 "$SCRIPT_DIR/state_manager.py" "$MODEL_DIR" --start "$1"
}

state_complete_phase() {
    python3 "$SCRIPT_DIR/state_manager.py" "$MODEL_DIR" --complete "$1"
}

state_fail_phase() {
    python3 "$SCRIPT_DIR/state_manager.py" "$MODEL_DIR" --fail "$1"
}

state_skip_phase() {
    python3 "$SCRIPT_DIR/state_manager.py" "$MODEL_DIR" --skip "$1"
}

state_print_status() {
    python3 "$SCRIPT_DIR/state_manager.py" "$MODEL_DIR" --status
}

state_can_resume() {
    python3 "$SCRIPT_DIR/state_manager.py" "$MODEL_DIR" --can-resume
}

state_get_resume_point() {
    python3 "$SCRIPT_DIR/state_manager.py" "$MODEL_DIR" --resume-point
}

state_get_phase_status() {
    python3 "$SCRIPT_DIR/state_manager.py" "$MODEL_DIR" --phase-status "$1" 2>/dev/null || echo "pending"
}

# Check if phase is already completed and skip if so
check_phase_completed() {
    local phase="$1"
    local status=$(state_get_phase_status "$phase")
    if [ "$status" = "completed" ]; then
        log_info "Phase '$phase' already completed, skipping..."
        return 0
    fi
    return 1
}

# ==============================================================================
# Phase Functions
# ==============================================================================

phase_init() {
    # Check if already completed (before printing header)
    if check_phase_completed "init"; then
        return 0
    fi
    
    log_step "1" "Initializing model directory"
    
    if [ -d "$MODEL_DIR" ]; then
        if [ -f "$MODEL_DIR/setup_state.json" ]; then
            # Check state to see where we left off
            STATE_STATUS=$(python3 "$SCRIPT_DIR/state_manager.py" "$MODEL_DIR" --can-resume 2>/dev/null || echo "false")
            if [ "$STATE_STATUS" = "True" ] || [ "$STATE_STATUS" = "true" ]; then
                log_info "Resuming from saved state"
            else
                log_info "Directory exists with completed state, continuing..."
            fi
        else
            log_info "Directory exists, initializing new state..."
        fi
    else
        if [ "$DRY_RUN" = true ]; then
            log_info "Would create: $MODEL_DIR"
        else
            mkdir -p "$MODEL_DIR"
            mkdir -p "$MODEL_DIR/logs"
            log_ok "Created: $MODEL_DIR"
        fi
    fi
    
    if [ "$DRY_RUN" != true ]; then
        state_init
        state_start_phase "init"
        
        # Copy device config from system config (not template)
        python3 -c "
import yaml
import json

system_config = yaml.safe_load(open('$SCRIPT_DIR/config/system.yaml'))
device = system_config.get('device', {})

device_config = {
    'device_name': device.get('name', 'unknown'),
    'description': device.get('description', ''),
    'cuda': {
        'version': device.get('cuda', {}).get('version', '12.9'),
        'home': device.get('cuda', {}).get('home', '/usr/local/cuda-12.9'),
    },
    'gpus': {
        'visible_devices': device.get('gpu', {}).get('visible_devices', '0'),
        'count': device.get('gpu', {}).get('count', 1),
        'names': device.get('gpu', {}).get('names', []),
        'memory_gb_per_gpu': device.get('gpu', {}).get('memory_gb', 80),
        'total_memory_gb': device.get('gpu', {}).get('count', 1) * device.get('gpu', {}).get('memory_gb', 80),
        'compute_capability': device.get('gpu', {}).get('compute_capability', '9.0'),
    },
    'system': {
        'total_ram_gb': device.get('system_ram_gb', 512),
        'cpu_count': device.get('cpu_count', 64),
    },
    'paths': system_config.get('paths', {}),
    'environment': system_config.get('environment', {}),
}

with open('$MODEL_DIR/device_config.json', 'w') as f:
    json.dump(device_config, f, indent=2)
print('Device config written')
"
        log_ok "Created device config from system config"
        
        state_complete_phase "init"
    fi
}

phase_fetch_info() {
    # Check if already completed
    if check_phase_completed "fetch_info"; then
        return 0
    fi
    
    log_step "2" "Fetching model information from HuggingFace"
    
    FETCH_SCRIPT="$SCRIPT_DIR/fetch_model_info.py"
    
    if [ ! -f "$FETCH_SCRIPT" ]; then
        log_error "fetch_model_info.py not found at $FETCH_SCRIPT"
        exit 1
    fi
    
    MODEL_INFO_FILE="$MODEL_DIR/model_info.json"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Would fetch model info for: $MODEL_ID"
    else
        state_start_phase "fetch_info"
        
        HF_HOME=$(python3 -c "import yaml; print(yaml.safe_load(open('$SCRIPT_DIR/config/system.yaml'))['paths']['hf_home'])" 2>/dev/null)
        if [ -z "$HF_HOME" ]; then
            error "Failed to load hf_home from config"
        fi
        
        python3 "$FETCH_SCRIPT" "$MODEL_ID" -o "$MODEL_INFO_FILE" --hf-home "$HF_HOME"
        
        if [ ! -f "$MODEL_INFO_FILE" ]; then
            state_fail_phase "fetch_info:Failed to fetch model info"
            log_error "Failed to fetch model information"
            exit 1
        fi
        
        # Check if model is vLLM compatible
        VLLM_COMPATIBLE=$(python3 -c "import json; print(json.load(open('$MODEL_INFO_FILE')).get('vllm_compatible', True))" 2>/dev/null)
        if [ "$VLLM_COMPATIBLE" = "False" ]; then
            log_warn "=============================================="
            log_warn "Model is NOT compatible with vLLM!"
            log_warn "GGUF models require llama.cpp instead."
            log_warn "Skipping remaining setup phases."
            log_warn "=============================================="
            state_complete_phase "fetch_info"
            state_skip_phase "download" "GGUF model - not vLLM compatible"
            state_skip_phase "venv" "GGUF model - not vLLM compatible"
            state_skip_phase "generate_config" "GGUF model - not vLLM compatible"
            state_skip_phase "optimize" "GGUF model - not vLLM compatible"
            state_complete_phase "finalize"
            exit 0
        fi
        
        state_complete_phase "fetch_info"
        log_ok "Model info saved to: $MODEL_INFO_FILE"
    fi
}

phase_download() {
    # Check if already completed
    if check_phase_completed "download"; then
        return 0
    fi
    
    if [ "$NO_DOWNLOAD" = true ]; then
        log_step "3" "Skipping model download (--no-download)"
        state_start_phase "download"
        state_skip_phase "download" "Skipped by user request"
        return
    fi
    
    log_step "3" "Downloading model from HuggingFace"
    
    DOWNLOAD_SCRIPT="$SCRIPT_DIR/download_model.py"
    
    if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
        log_error "download_model.py not found at $DOWNLOAD_SCRIPT"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Would download model: $MODEL_ID"
        return
    fi
    
    state_start_phase "download"
    
    HF_HOME=$(python3 -c "import yaml; print(yaml.safe_load(open('$SCRIPT_DIR/config/system.yaml'))['paths']['hf_home'])" 2>/dev/null)
    if [ -z "$HF_HOME" ]; then
        error "Failed to load hf_home from config"
    fi
    
    DOWNLOAD_ARGS="$MODEL_ID --model-dir $MODEL_DIR --hf-home $HF_HOME"
    
    if [ "$TMUX_DOWNLOAD" = true ]; then
        DOWNLOAD_ARGS="$DOWNLOAD_ARGS --tmux"
    fi
    
    if python3 "$DOWNLOAD_SCRIPT" $DOWNLOAD_ARGS; then
        state_complete_phase "download"
        log_ok "Model downloaded successfully"
    else
        if [ "$TMUX_DOWNLOAD" = true ]; then
            log_warn "Download may still be in progress (check tmux session)"
        else
            log_error "Download failed"
            # Try agentic recovery with OpenCode
            log_info "Attempting agentic recovery..."
            ERROR_MSG="Download failed for $MODEL_ID"
            if python3 "$SCRIPT_DIR/opencode_phase_agent.py" "$MODEL_DIR" "download" "$ERROR_MSG"; then
                log_ok "OpenCode agent recovery succeeded"
            else
                log_error "OpenCode agent could not fix the issue"
                exit 1
            fi
        fi
    fi
}

phase_venv() {
    # Check if already completed
    if check_phase_completed "venv"; then
        return 0
    fi
    
    if [ "$NO_VENV" = true ] || [ "$NO_INSTALL" = true ]; then
        log_step "4" "Skipping virtual environment (--no-venv or --no-install)"
        state_start_phase "venv"
        state_skip_phase "venv" "Skipped by user request"
        return
    fi
    
    log_step "4" "Creating virtual environment"
    
    VENV_SCRIPT="$SCRIPT_DIR/create_venv.sh"
    
    if [ ! -f "$VENV_SCRIPT" ]; then
        log_error "create_venv.sh not found at $VENV_SCRIPT"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Would create venv in: $MODEL_DIR/.venv"
        return
    fi
    
    state_start_phase "venv"
    
    VENV_ARGS=("$MODEL_DIR")
    [ -n "$PYTHON_VERSION" ] && VENV_ARGS+=("$PYTHON_VERSION")
    [ "$VLLM_VERSION" != "latest" ] && VENV_ARGS+=("$VLLM_VERSION")
    
    if bash "$VENV_SCRIPT" "${VENV_ARGS[@]}"; then
        state_complete_phase "venv"
        log_ok "Virtual environment created"
    else
        state_fail_phase "venv:Failed to create virtual environment"
        log_error "Virtual environment creation failed"
        
        # Try agentic recovery with OpenCode
        log_info "Attempting agentic recovery for venv..."
        ERROR_MSG="Virtual environment creation failed"
        if python3 "$SCRIPT_DIR/opencode_phase_agent.py" "$MODEL_DIR" "venv" "$ERROR_MSG"; then
            log_info "OpenCode agent recovery succeeded, retrying venv creation..."
            if bash "$VENV_SCRIPT" "${VENV_ARGS[@]}"; then
                state_complete_phase "venv"
                log_ok "Virtual environment created after agent fix"
            else
                log_error "Failed to create virtual environment even after agent recovery"
                exit 1
            fi
        else
            log_error "OpenCode agent could not fix the issue"
            exit 1
        fi
    fi
    
    # Phase 4.1: Smoke test
    phase_smoke_test
}

phase_smoke_test() {
    log_step "4.1" "Running vLLM smoke test"
    
    # Check if already completed
    if check_phase_completed "smoke_test"; then
        log_info "Smoke test already completed, skipping..."
        return 0
    fi
    
    # Check if venv exists
    if [ ! -d "$MODEL_DIR/.venv" ]; then
        log_warn "Virtual environment not found, skipping smoke test"
        state_start_phase "smoke_test"
        state_skip_phase "smoke_test" "No virtual environment"
        return 0
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Would run vLLM smoke test"
        return 0
    fi
    
    state_start_phase "smoke_test"
    
    # Load smoke test settings from system.yaml
    local SYSTEM_CONFIG="$SCRIPT_DIR/config/system.yaml"
    local SMOKE_CONFIG="$MODEL_DIR/smoke_test_config.yaml"
    
    # If model doesn't have smoke config, create it from system.yaml
    if [ ! -f "$SMOKE_CONFIG" ]; then
        python3 -c "
import yaml
import sys

# Load from system.yaml
system = yaml.safe_load(open('$SYSTEM_CONFIG'))
smoke = system.get('smoke_test', {})

# Create model-specific config with vllm_args
config = {
    'smoke_test': {
        'vllm_args': smoke.get('vllm_args', {
            'max_model_len': 1024,
            'tensor_parallel_size': 1,
            'dtype': 'auto',
            'trust_remote_code': True,
            'enable_chunked_prefill': True
        }),
        'timeout_seconds': smoke.get('timeout_seconds', 1800),
        'poll_interval_seconds': smoke.get('poll_interval_seconds', 2),
        'max_retries': smoke.get('max_retries', 3),
        'start_port': smoke.get('start_port', 29500),
        'enable_agentic_recovery': True
    }
}

yaml.dump(config, open('$SMOKE_CONFIG', 'w'), default_flow_style=False)
print('Created smoke_test_config.yaml')
" 2>/dev/null
        log_info "Created $SMOKE_CONFIG from system.yaml"
    fi
    
    # Load settings from model's smoke test config
    local SMOKE_TIMEOUT=$(python3 -c "import yaml; print(yaml.safe_load(open('$SMOKE_CONFIG')).get('smoke_test', {}).get('timeout_seconds', 1800))" 2>/dev/null || echo 1800)
    local SMOKE_MAX_RETRIES=$(python3 -c "import yaml; print(yaml.safe_load(open('$SMOKE_CONFIG')).get('smoke_test', {}).get('max_retries', 3))" 2>/dev/null || echo 3)
    local SMOKE_POLL_INTERVAL=$(python3 -c "import yaml; print(yaml.safe_load(open('$SMOKE_CONFIG')).get('smoke_test', {}).get('poll_interval_seconds', 2))" 2>/dev/null || echo 2)
    local SMOKE_START_PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('$SMOKE_CONFIG')).get('smoke_test', {}).get('start_port', 29500))" 2>/dev/null || echo 29500)
    
    # Build vLLM args from config file using intelligent flag validator
    log_info "Validating vLLM flags using runtime introspection..."
    
    local VLLM_ARGS=""
    local VALIDATION_OUTPUT=""
    local VALIDATION_JSON=""
    
    # Run flag validator to get validated args
    VALIDATION_OUTPUT=$(python3 "$SCRIPT_DIR/vllm_flag_validator.py" \
        --model-dir "$MODEL_DIR" \
        --config-file "$SMOKE_CONFIG" \
        --format json 2>&1)
    
    if [ $? -eq 0 ]; then
        VALIDATION_JSON="$VALIDATION_OUTPUT"
        # Extract valid_args from JSON
        VLLM_ARGS=$(echo "$VALIDATION_JSON" | python3 -c "import sys,json; print(' '.join(json.load(sys.stdin)['valid_args']))" 2>/dev/null)
        
        # Log removed flags if any
        local REMOVED_COUNT=$(echo "$VALIDATION_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['removed_flags']))" 2>/dev/null || echo 0)
        if [ "$REMOVED_COUNT" -gt 0 ]; then
            log_warn "Flag validation removed $REMOVED_COUNT invalid flag(s):"
            echo "$VALIDATION_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for item in data.get('removed_flags', []):
    print(f\"  - --{item['flag'].replace('_', '-')}: {item['reason']}\")
" 2>/dev/null
        fi
        
        # Log warnings if any
        local WARNINGS=$(echo "$VALIDATION_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print('\\n'.join(d.get('warnings', [])))" 2>/dev/null)
        if [ -n "$WARNINGS" ]; then
            log_warn "Flag validation warnings:"
            echo "$WARNINGS" | while read line; do
                [ -n "$line" ] && log_warn "  $line"
            done
        fi
    else
        log_warn "Flag validator failed, falling back to basic conversion"
        VLLM_ARGS=$(python3 -c "
import yaml
c = yaml.safe_load(open('$SMOKE_CONFIG'))
args = c.get('smoke_test', {}).get('vllm_args', {})
arg_list = []
for k, v in args.items():
    flag = f'--{k.replace(\"_\", \"-\")}'
    if isinstance(v, bool):
        if v:
            arg_list.append(flag)
        else:
            arg_list.append(f'--no-{k.replace(\"_\", \"-\")}')
    elif v == \"\" or v is None:
        arg_list.append(flag)
    else:
        arg_list.append(f'{flag} {v}')
print(' '.join(arg_list))
" 2>/dev/null)
    fi
    
    log_info "Smoke test config: $SMOKE_CONFIG"
    log_info "vLLM args: $VLLM_ARGS"
    
    log_info "Running vLLM serve smoke test (timeout: ${SMOKE_TIMEOUT}s)..."
    
    # Get model ID
    local MODEL_ID=""
    if [ -f "$MODEL_DIR/model_info.json" ] && command -v jq &> /dev/null; then
        MODEL_ID=$(jq -r '.model_id // empty' "$MODEL_DIR/model_info.json")
    fi
    
    if [ -z "$MODEL_ID" ]; then
        log_warn "Model ID not found, skipping smoke test"
        state_skip_phase "smoke_test" "No model ID"
        return 0
    fi
    
    # Find free port
    local TEST_PORT=$SMOKE_START_PORT
    while netstat -tuln 2>/dev/null | grep -q ":$TEST_PORT "; do
        TEST_PORT=$((TEST_PORT + 1))
    done
    
    log_info "Using test port: $TEST_PORT"

    local VLLM_BIN="$MODEL_DIR/.venv/bin/vllm"
    local SMOKE_LOG="$MODEL_DIR/vllm_smoke_test.log"
    local ATTEMPT=1
    local HEALTHY=false
    local SMOKE_ITERATIONS=$((SMOKE_TIMEOUT / SMOKE_POLL_INTERVAL))

    local VISIBLE_DEVICES=$(python3 -c "import json; d=json.load(open('$MODEL_DIR/device_config.json')); print(d.get('gpus', {}).get('visible_devices', '0,1,2,3'))" 2>/dev/null || echo "0,1,2,3")
    export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES"
    log_info "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    while [ $ATTEMPT -le $SMOKE_MAX_RETRIES ] && [ "$HEALTHY" = false ]; do
        if [ $ATTEMPT -gt 1 ]; then
            log_info "Retry attempt $ATTEMPT/$SMOKE_MAX_RETRIES..."
            TEST_PORT=$((TEST_PORT + 1))
            log_info "Using new test port: $TEST_PORT"
        fi

        "$VLLM_BIN" serve "$MODEL_ID" $VLLM_ARGS --port $TEST_PORT > "$SMOKE_LOG" 2>&1 &
        local VLLM_PID=$!
        
        # Wait for health
        echo -n "  Polling health (every ${SMOKE_POLL_INTERVAL}s): "
        for i in $(seq 1 $SMOKE_ITERATIONS); do
            sleep $SMOKE_POLL_INTERVAL
            echo -n "."
            
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo ""
                log_warn "vLLM process died during startup"
                break
            fi
            
            if curl -s "http://localhost:$TEST_PORT/health" > /dev/null 2>&1; then
                echo " ✓"
                HEALTHY=true
                break
            fi
        done
        
        if [ "$HEALTHY" = false ]; then
            echo " ✗"
            log_warn "Health check timeout - killing vLLM process"
        fi
        
        # Cleanup - kill vLLM process forcefully if still running
        if kill -0 $VLLM_PID 2>/dev/null; then
            kill $VLLM_PID 2>/dev/null || true
            sleep 2
            if kill -0 $VLLM_PID 2>/dev/null; then
                log_warn "Process didn't terminate, using SIGKILL"
                kill -9 $VLLM_PID 2>/dev/null || true
            fi
        fi
        wait $VLLM_PID 2>/dev/null || true
        
        # Try agentic recovery if failed
        if [ "$HEALTHY" = false ] && [ $ATTEMPT -lt $SMOKE_MAX_RETRIES ]; then
            log_info "Attempting OpenCode agent recovery for smoke test..."
            log_info "Agent can modify: $MODEL_DIR/smoke_test_config.yaml"
            local ERROR_LOG=$(tail -n 100 "$SMOKE_LOG" 2>/dev/null || echo "No logs")
            if python3 "$SCRIPT_DIR/opencode_phase_agent.py" "$MODEL_DIR" "smoke_test" "$ERROR_LOG"; then
                log_info "Agent applied fix, reloading config..."
                # Reload and revalidate config in case agent modified it
                VLLM_ARGS=$(python3 "$SCRIPT_DIR/vllm_flag_validator.py" \
                    --model-dir "$MODEL_DIR" \
                    --config-file "$SMOKE_CONFIG" \
                    --format json 2>/dev/null | python3 -c "import sys,json; print(' '.join(json.load(sys.stdin)['valid_args']))" 2>/dev/null)
                log_info "Updated vLLM args: $VLLM_ARGS"
            else
                log_warn "Agent could not fix automatically"
            fi
        fi
        
        ATTEMPT=$((ATTEMPT + 1))
    done
    
    if [ "$HEALTHY" = true ]; then
        log_ok "Smoke test passed"
        state_complete_phase "smoke_test"
        rm -f "$SMOKE_LOG"  # Clean up on success
    else
        log_warn "Smoke test failed after $SMOKE_MAX_RETRIES attempts"
        log_warn "Check $SMOKE_LOG for details"
        state_fail_phase "smoke_test:Failed after $SMOKE_MAX_RETRIES attempts"
        # Continue anyway - agent will try to fix during optimization
    fi
    
    # Run vLLM config extraction after smoke test
    phase_extract_vllm_config
}

phase_extract_vllm_config() {
    # Check if already completed
    if check_phase_completed "extract_vllm_config"; then
        return 0
    fi
    
    log_step "5" "Extracting vLLM configuration from model"
    
    EXTRACTOR_SCRIPT="$SCRIPT_DIR/extract_vllm_config.py"
    
    if [ ! -f "$EXTRACTOR_SCRIPT" ]; then
        log_warn "extract_vllm_config.py not found, skipping extraction"
        state_start_phase "extract_vllm_config"
        state_skip_phase "extract_vllm_config" "Extractor script not found"
        return
    fi
    
    # Check if venv exists
    if [ ! -d "$MODEL_DIR/.venv" ]; then
        log_warn "Virtual environment not found, skipping extraction"
        state_start_phase "extract_vllm_config"
        state_skip_phase "extract_vllm_config" "No virtual environment"
        return
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Would extract vLLM config to: $MODEL_DIR/vllm_extracted_config.json"
        return
    fi
    
    state_start_phase "extract_vllm_config"
    
    log_info "Running vLLM introspection (this may take a minute)..."
    
    # Run extractor using model's venv
    if "$MODEL_DIR/.venv/bin/python" "$EXTRACTOR_SCRIPT" "$MODEL_DIR" --max-model-len 1024; then
        if [ -f "$MODEL_DIR/vllm_extracted_config.json" ]; then
            log_ok "vLLM config extracted successfully"
            state_complete_phase "extract_vllm_config"
        else
            log_warn "Extraction completed but output file not found"
            state_fail_phase "extract_vllm_config:Output file missing"
        fi
    else
        log_warn "vLLM config extraction failed (model may not be supported)"
        state_fail_phase "extract_vllm_config:Extraction failed"
        # Continue anyway - agent will fall back to guessing
    fi
    
    # Also extract available flags for intelligent optimization
    log_info "Extracting available vLLM flags..."
    FLAGS_SCRIPT="$SCRIPT_DIR/extract_vllm_flags.py"
    if [ -f "$FLAGS_SCRIPT" ]; then
        if "$MODEL_DIR/.venv/bin/python" "$FLAGS_SCRIPT" "$MODEL_DIR" --profile balanced; then
            if [ -f "$MODEL_DIR/vllm_available_flags.json" ]; then
                log_ok "vLLM flags extracted successfully"
            else
                log_warn "Flag extraction completed but output file not found"
            fi
        else
            log_warn "vLLM flag extraction failed (continuing anyway)"
        fi
    else
        log_warn "extract_vllm_flags.py not found, skipping flag extraction"
    fi
}

phase_optimize() {
    # OPTIMIZE PHASE ALWAYS RUNS when explicitly requested (--optimize or --research)
    # This allows experimenting with different configurations
    
    # Check if optimization is enabled
    if [ "$OPTIMIZE" != true ] && [ "$RESEARCH" != true ]; then
        log_info "Optimization disabled (use --optimize to enable)"
        return
    fi
    
    # Always skip state check for optimize - we want to run it every time it's requested
    # (unless force flag is explicitly used to override)
    if [ "$OPTIMIZE_FORCE" != true ]; then
        local status=$(state_get_phase_status "optimize")
        if [ "$status" = "completed" ]; then
            log_info "Previous optimization found - running again to test new configurations"
        fi
    fi
    
    log_step "6" "Optimizing vLLM configuration"
    
    OPTIMIZE_SCRIPT="$SCRIPT_DIR/optimize_vllm_config.py"
    
    if [ ! -f "$OPTIMIZE_SCRIPT" ]; then
        log_warn "optimize_vllm_config.py not found at $OPTIMIZE_SCRIPT"
        log_info "Skipping optimization phase"
        return
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Would run vLLM optimization agent for: $MODEL_ID"
        log_info "  Progressive targets from config"
        log_info "  Max retries from config"
        return
    fi
    
    state_start_phase "optimize"
    
    log_info "Running VLLM auto-optimization agent..."
    log_info "Testing progressive context targets from config"
    
    OPTIMIZE_ARGS="--model-dir $MODEL_DIR --verbose"
    if [ "$OPTIMIZE_FORCE" = true ]; then
        OPTIMIZE_ARGS="$OPTIMIZE_ARGS --force"
        log_info "Force flag set: will re-optimize even if already optimized"
    fi
    if [ "$OPTIMIZE_AGENTIC" = true ]; then
        OPTIMIZE_ARGS="$OPTIMIZE_ARGS --agentic"
        log_info "Using OpenCode-based agentic optimization"
    fi
    
    # Pass deep mode flag if enabled
    if [ "$DEEP_MODE" = true ]; then
        OPTIMIZE_ARGS="$OPTIMIZE_ARGS --deep"
        log_info "Deep optimization mode enabled (thorough backend/CUDA graph research)"
    fi
    
    if python3 "$OPTIMIZE_SCRIPT" $OPTIMIZE_ARGS; then
        state_complete_phase "optimize"
        log_ok "Configuration optimized successfully"
        
        # Check results
        if [ -f "$MODEL_DIR/optimized_config.json" ]; then
            BEST_CONTEXT=$(jq -r '.best_context // 0' "$MODEL_DIR/optimized_config.json")
            if [ "$BEST_CONTEXT" -gt 0 ]; then
                log_ok "Best context achieved: ${BEST_CONTEXT} tokens"
            fi
        fi
    else
        log_warn "Optimization completed with warnings or errors"
        state_complete_phase "optimize"
    fi
}

phase_generate_config() {
    log_step "5" "Generating PM2 configuration and management script"
    
    PM2_SCRIPT="$SCRIPT_DIR/generate_pm2_config.py"
    MODEL_SH_SCRIPT="$SCRIPT_DIR/generate_model_sh.py"
    
    if [ ! -f "$PM2_SCRIPT" ]; then
        log_error "generate_pm2_config.py not found at $PM2_SCRIPT"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Would generate PM2 config at: $MODEL_DIR/ecosystem.config.js"
        log_info "Would generate model.sh at: $MODEL_DIR/model.sh"
        log_info "Would copy documentation files from checkpoint"
        return
    fi
    
    state_start_phase "generate_config"
    
    # Generate PM2 config with --copy-docs to also copy README, LICENSE, etc.
    python3 "$PM2_SCRIPT" "$MODEL_DIR" --port $PORT --device-config "$MODEL_DIR/device_config.json" --copy-docs
    
    if [ ! -f "$MODEL_DIR/ecosystem.config.js" ]; then
        state_fail_phase "generate_config:Failed to generate PM2 config"
        log_error "Failed to generate PM2 configuration"
        exit 1
    fi
    
    log_ok "PM2 config saved"
    
    # Check if documentation was copied
    if ls "$MODEL_DIR"/*.md "$MODEL_DIR"/*.txt >/dev/null 2>&1; then
        log_ok "Documentation files copied from checkpoint"
    fi
    
    if [ -f "$MODEL_SH_SCRIPT" ]; then
        python3 "$MODEL_SH_SCRIPT" "$MODEL_DIR"
        chmod +x "$MODEL_DIR/model.sh"
        log_ok "Management script saved"
    fi
    
    state_complete_phase "generate_config"
}

phase_finalize() {
    log_step "7" "Finalizing setup"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Would create README and finalize"
        return
    fi
    
    state_start_phase "finalize"
    
    if [ -f "$MODEL_DIR/model_info.json" ] && command -v jq &> /dev/null; then
        MODEL_ARCH=$(jq -r '.architecture // "unknown"' "$MODEL_DIR/model_info.json")
        MODEL_FAMILY=$(jq -r '.family // "unknown"' "$MODEL_DIR/model_info.json")
        MODEL_CONTEXT=$(jq -r '.context_length // 4096' "$MODEL_DIR/model_info.json")
        MODEL_PARAMS=$(jq -r '.parameter_count // "unknown"' "$MODEL_DIR/model_info.json")
    else
        MODEL_ARCH="unknown"
        MODEL_FAMILY="unknown"
        MODEL_CONTEXT="unknown"
        MODEL_PARAMS="unknown"
    fi
    
    cat > "$MODEL_DIR/README.md" << EOF
# $MODEL_NAME

**Model ID:** \`$MODEL_ID\`

## Model Information

| Property | Value |
|----------|-------|
| Architecture | $MODEL_ARCH |
| Family | $MODEL_FAMILY |
| Parameters | $MODEL_PARAMS |
| Context Length | $MODEL_CONTEXT |

## Quick Start

\`\`\`bash
# Start the server
./model.sh start

# Check status
./model.sh status

# View logs
./model.sh logs

# Stop the server
./model.sh stop
\`\`\`

## Server Endpoints

- **Base URL:** \`http://localhost:$PORT\`
- **OpenAI API:** \`http://localhost:$PORT/v1\`
- **Health:** \`http://localhost:$PORT/health\`

## Directory Structure

\`\`\`
$MODEL_NAME/
├── .venv/              # Virtual environment
├── ecosystem.config.js # PM2 configuration
├── model.sh            # Management script
├── model_info.json     # Model metadata
├── optimized_config.json # Optimized vLLM config
├── device_config.json  # Device-level config
├── logs/               # Server logs
└── README.md           # This file
\`\`\`

---
Generated by Auto Model Deployer on $(date +%Y-%m-%d)
EOF
    
    state_complete_phase "finalize"
    log_ok "README created"
    
    # Copy deploy.py for profile-based deployment
    DEPLOY_SCRIPT="$SCRIPT_DIR/deploy.py"
    if [ -f "$DEPLOY_SCRIPT" ]; then
        cp "$DEPLOY_SCRIPT" "$MODEL_DIR/deploy.py"
        chmod +x "$MODEL_DIR/deploy.py"
        log_ok "Deploy script copied"
    fi
    
    state_complete_phase "finalize"
}

# ==============================================================================
# Argument Parsing
# ==============================================================================

parse_args() {
    HF_URL=""
    BASE_DIR="$DEFAULT_BASE_DIR"
    PORT="$DEFAULT_PORT"
    PYTHON_VERSION=""
    VLLM_VERSION="latest"
    TENSOR_PARALLEL=""
    MAX_MODEL_LEN=""
    NO_VENV=false
    NO_INSTALL=false
    NO_DOWNLOAD=false
    TMUX_DOWNLOAD=false
    RESEARCH=false
    OPTIMIZE=true
    OPTIMIZE_AGENTIC=true
    DEEP_MODE=false
    OPTIMIZE_FORCE=false
    DEPLOY=false
    DEPLOY_COMMAND=""
    DEPLOY_PROFILE="balanced"
    RESUME=false
    STATUS_ONLY=false
    DRY_RUN=false
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --base-dir)
                BASE_DIR="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --python)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            --vllm)
                VLLM_VERSION="$2"
                shift 2
                ;;
            --tensor-parallel)
                TENSOR_PARALLEL="$2"
                shift 2
                ;;
            --max-model-len)
                MAX_MODEL_LEN="$2"
                shift 2
                ;;
            --no-venv)
                NO_VENV=true
                shift
                ;;
            --no-install)
                NO_INSTALL=true
                shift
                ;;
            --no-download)
                NO_DOWNLOAD=true
                shift
                ;;
            --tmux-download)
                TMUX_DOWNLOAD=true
                shift
                ;;
            --research)
                RESEARCH=true
                shift
                ;;
            --optimize)
                OPTIMIZE=true
                shift
                ;;
            --agentic)
                OPTIMIZE_AGENTIC=true
                shift
                ;;
            --deep)
                DEEP_MODE=true
                shift
                ;;
            --force)
                OPTIMIZE_FORCE=true
                shift
                ;;
            --deploy)
                DEPLOY=true
                if [ -n "$2" ] && [[ "$2" == @(start|stop|restart) ]]; then
                    DEPLOY_COMMAND="$2"
                    shift 2
                else
                    DEPLOY_COMMAND="start"
                    shift
                fi
                ;;
            --mode)
                if [ -n "$2" ] && [[ "$2" == @(speed|balanced|quality) ]]; then
                    DEPLOY_PROFILE="$2"
                    shift 2
                else
                    log_error "--mode requires a profile: speed, balanced, or quality"
                    exit 1
                fi
                ;;
            --resume)
                RESUME=true
                shift
                ;;
            --status)
                STATUS_ONLY=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                if [ -z "$HF_URL" ]; then
                    HF_URL="$1"
                else
                    log_error "Unexpected argument: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate required arguments
    if [ -z "$HF_URL" ]; then
        log_error "Model ID or URL is required"
        show_usage
        exit 1
    fi
    
    # Parse model ID from URL or use directly
    if [[ "$HF_URL" == http* ]]; then
        MODEL_ID=$(echo "$HF_URL" | sed 's|.*huggingface.co/||' | sed 's|/tree/main||' | sed 's|/blob/main||')
    else
        MODEL_ID="$HF_URL"
    fi
    
    # Create model name from ID
    MODEL_NAME=$(echo "$MODEL_ID" | sed 's|/|-|g' | tr '[:upper:]' '[:lower:]')
    MODEL_DIR="$BASE_DIR/$MODEL_NAME"
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    parse_args "$@"
    
    print_banner
    
    log_info "Model ID: $MODEL_ID"
    log_info "Model directory: $MODEL_DIR"
    
    check_command python3
    check_command pip
    check_command pm2
    
    if ! command -v jq &> /dev/null; then
        log_warn "jq not found - some features may be limited"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_warn "DRY RUN MODE - No changes will be made"
        echo ""
    fi
    
    if [ "$RESUME" = true ] && [ -f "$MODEL_DIR/setup_state.json" ]; then
        log_info "Resuming from previous setup..."
        RESUME_POINT=$(state_get_resume_point)
        log_info "Resume point: $RESUME_POINT"
    fi
    
    phase_init
    phase_fetch_info
    phase_download
    phase_venv
    phase_smoke_test
    phase_extract_vllm_config
    phase_generate_config
    phase_optimize
    phase_finalize
    
    # Deploy if requested
    if [ "$DEPLOY" = true ]; then
        log_step "8" "Deploying with PM2"
        if [ -f "$MODEL_DIR/deploy.py" ]; then
            log_info "Starting deployment: $DEPLOY_COMMAND with $DEPLOY_PROFILE profile"
            cd "$MODEL_DIR" && python3 deploy.py "$DEPLOY_COMMAND" "$DEPLOY_PROFILE"
        else
            log_warn "deploy.py not found, skipping deployment"
        fi
    fi
    
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                 Setup Complete!                              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Model Directory:${NC} $MODEL_DIR"
    echo ""
    echo -e "${BOLD}Files Created:${NC}"
    echo "  - model_info.json       (Model metadata)"
    echo "  - device_config.json    (Device configuration)"
    echo "  - optimized_config.json (Optimized vLLM config)"
    echo "  - ecosystem.config.js   (PM2 configuration)"
    echo "  - model.sh              (Management script)"
    echo "  - deploy.py             (Profile-based deployment)"
    echo "  - README.md             (Documentation)"
    [ "$NO_VENV" = false ] && [ "$NO_INSTALL" = false ] && echo "  - .venv/                (Virtual environment)"
    echo ""
    echo -e "${BOLD}Next Steps:${NC}"
    echo ""
    echo "  1. Navigate to model directory:"
    echo "     ${CYAN}cd $MODEL_DIR${NC}"
    echo ""
    if [ "$DEPLOY" = true ]; then
        echo "  2. Server is already running with $DEPLOY_PROFILE profile"
        echo "     ${CYAN}./deploy.py status${NC}"
    else
        echo "  2. Start the server:"
        echo "     ${CYAN}./deploy.py start${NC}  # Uses balanced profile (default)"
        echo "     ${CYAN}./deploy.py start speed${NC}  # Uses speed profile"
        echo "     ${CYAN}./deploy.py start quality${NC}  # Uses quality profile"
    fi
    echo ""
    echo "  3. Test the API:"
    echo "     ${CYAN}curl http://localhost:$PORT/v1/models${NC}"
    echo ""
}

main "$@"
