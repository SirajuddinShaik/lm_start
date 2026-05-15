# 🚀 lm-start

**One-command vLLM model deployment with intelligent OpenCode integration.**

lm-start automates the entire pipeline of deploying Large Language Models locally using vLLM - from hardware detection to optimized configuration generation.

[LLM Inference with vLLM - LinkedIn Post with results](https://www.linkedin.com/posts/sirajuddin-shaik-_llm-inference-vllm-ugcPost-7460782617676804096-saJb?utm_source=share&utm_medium=member_desktop&rcm=ACoAADxng8QBcH6qX8zhibMl0YlMhWvH3xjWH7E)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-integrated-orange.svg)](https://github.com/vllm-project/vllm)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands Reference](#commands-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Security](#security)

---

## 🎯 Overview

lm-start transforms the complex process of setting up vLLM models into a simple, automated workflow:

1. **Initialize** - Detects your hardware (GPU, CPU, RAM, CUDA) and sets up the environment
2. **Configure** - Manages credentials securely (HF_TOKEN, etc.)
3. **Deploy** - Downloads models, creates optimized vLLM configurations, and sets up PM2 process management
4. **Monitor** - Track deployment status and view logs in real-time

Perfect for developers who want to run LLMs locally without wrestling with configuration files.

---

## ✨ Features

### 🔧 Automated Setup
- **Hardware Auto-Detection**: Automatically detects GPUs (NVIDIA), CPU cores, RAM, and CUDA version
- **OpenCode Integration**: Automatically installs OpenCode v1.4.0 for AI-assisted optimization
- **Smart Defaults**: Generates optimized vLLM configurations based on your hardware

### 🎛️ Model Management
- **One-Command Deployment**: `lm-start setup <model>` handles everything
- **Profile Selection**: Choose between `speed`, `balanced`, or `quality` optimization profiles
- **Resume Capability**: Interrupted deployments can be resumed from any phase
- **9-Phase Pipeline**: init → fetch_info → download → venv → smoke_test → extract_config → generate_config → optimize → finalize

### 🔐 Secure Credential Management
- **Encrypted Storage**: Credentials stored in `~/.lm-start/config/credentials.yaml`
- **Masked Display**: Sensitive values are masked when listing (e.g., `hf_t********`)
- **Environment Export**: Credentials automatically exported when running commands
- **Easy Management**: Add, list, and remove credentials via CLI

### 📊 Monitoring & Diagnostics
- **Real-time Status**: Check which models are deployed and their resource usage
- **Log Streaming**: View logs in real-time with `lm-start logs <model> -f`
- **System Health**: Built-in `doctor` command diagnoses common issues
- **PM2 Integration**: Seamless integration with PM2 process manager

### 🛠️ Developer Experience
- **Rich CLI Output**: Colorful, formatted output with progress indicators
- **Comprehensive Help**: Every command has detailed `--help`
- **Error Handling**: Graceful degradation and helpful error messages
- **Configuration Validation**: Validates setups before attempting deployment

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Linux system with NVIDIA GPU (recommended)
- Internet connection for model downloads

### Install from source

```bash
cd /home/siraj/Documents/lm_start
pip install -e .
```

### Verify Installation

```bash
lm-start --version
# Output: lm-start version 0.1.0
```

---

## 🚀 Quick Start

### 1. Initialize lm-start

```bash
lm-start init
```

This will:
- Create `~/.lm-start/` directory structure
- Install OpenCode v1.4.0
- Detect your hardware (GPU, CPU, RAM, CUDA)
- Generate `system.yaml` with your hardware profile

### 2. Add Credentials (for gated models)

```bash
# Add HuggingFace token
lm-start env add HF_TOKEN "your_hf_token_here"

# Add from current environment variable
lm-start env add HF_TOKEN --from-env

# List stored credentials (values masked)
lm-start env list

# Remove a credential
lm-start env remove HF_TOKEN
```

### 3. Deploy a Model

```bash
# Basic deployment
lm-start setup meta-llama/Llama-2-7b-chat-hf

# With optimization profile
lm-start setup Qwen/Qwen2.5-32B --profile speed

# With automatic optimization
lm-start setup microsoft/Phi-3-mini-4k-instruct --optimize

# Resume interrupted setup
lm-start setup meta-llama/Llama-2-7b-chat-hf --resume
```

Available profiles:
- `speed` - Maximize throughput
- `balanced` - Balance speed and quality (default)
- `quality` - Maximize generation quality

### 4. Monitor Deployment

```bash
# Check all deployed models
lm-start status

# Check specific model
lm-start status Llama-2-7b-chat-hf

# View logs
lm-start logs Llama-2-7b-chat-hf

# Follow logs in real-time
lm-start logs Llama-2-7b-chat-hf --follow

# Or shorter
lm-start logs Llama-2-7b-chat-hf -f

# Show last N lines
lm-start logs Llama-2-7b-chat-hf --lines 50
```

### 5. System Diagnostics

```bash
lm-start doctor
```

Checks:
- ✅ OpenCode installation
- ✅ Hardware detection (GPU, CPU, RAM)
- ✅ Credentials configuration
- ✅ Python dependencies
- ✅ NVIDIA driver availability

---

## 📚 Commands Reference

### Global Options

```bash
lm-start [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion    Install completion for the current shell
  --show-completion       Show completion for the current shell
  --help                  Show this message and exit
```

### `init` - Initialize Configuration

```bash
lm-start init [OPTIONS]

Initialize lm-start configuration directory.

Creates:
  ~/.lm-start/              # Main config directory
  ~/.lm-start/config/       # Configuration files
  ~/.lm-start/opencode/     # OpenCode installation

Options:
  --help    Show this message and exit
```

### `env` - Manage Environment Variables

```bash
lm-start env [OPTIONS] COMMAND [ARGS]...

Manage environment variables and credentials securely.

Commands:
  add      Add a new environment variable
  list     List all stored credentials (masked)
  remove   Remove an environment variable
```

#### `env add`

```bash
lm-start env add [OPTIONS] KEY VALUE

Add a new credential.

Arguments:
  KEY    Variable name (e.g., HF_TOKEN)
  VALUE  Variable value

Options:
  --from-env    Read value from current environment
  --help        Show this message and exit

Examples:
  lm-start env add HF_TOKEN "hf_xxx"
  lm-start env add WANDB_API_KEY "xxx" --from-env
```

#### `env list`

```bash
lm-start env list [OPTIONS]

List all stored credentials (values masked for security).

Example output:
  ╭──────────────────╮
  │ Stored Credentials    │
  ├──────────┬────────────┤
  │ Key      │ Value      │
  ├──────────┼────────────┤
  │ HF_TOKEN │ hf_t****** │
  ╰──────────┴────────────╯
```

#### `env remove`

```bash
lm-start env remove [OPTIONS] KEY

Remove a stored credential.

Arguments:
  KEY    Variable name to remove
```

### `setup` - Deploy Model

```bash
lm-start setup [OPTIONS] MODEL

Setup model deployment with vLLM.

Arguments:
  * MODEL    Model name to setup (e.g., Qwen/Qwen2.5-32B) [required]

Options:
  -p, --profile TEXT   Optimization profile: speed|balanced|quality [default: balanced]
  -o, --optimize       Run optimization to find best configuration
  -r, --resume         Resume from last completed phase
  --help               Show this message and exit

Examples:
  lm-start setup meta-llama/Llama-2-7b-chat-hf
  lm-start setup Qwen/Qwen2.5-32B --profile speed
  lm-start setup microsoft/Phi-3-mini-4k-instruct --optimize
  lm-start setup meta-llama/Llama-2-7b-chat-hf --resume
```

**Deployment Phases:**
1. **init** - Directory setup
2. **fetch_info** - Get model metadata from HuggingFace
3. **download** - Download model weights
4. **venv** - Create Python virtualenv with vLLM
5. **smoke_test** - Test vLLM startup
6. **extract_config** - Get runtime configuration
7. **generate_config** - Create PM2 configuration
8. **optimize** - Run AI optimization (if --optimize)
9. **finalize** - Create README and management scripts

### `status` - Check Deployment Status

```bash
lm-start status [OPTIONS] [MODEL]

Show deployment status of models.

Arguments:
  MODEL    Model name to show status for (optional)

Options:
  --help    Show this message and exit

Examples:
  lm-start status           # List all models
  lm-start status Llama-2   # Show specific model
```

### `logs` - View Deployment Logs

```bash
lm-start logs [OPTIONS] MODEL

Show deployment logs for a model.

Arguments:
  * MODEL    Model name to show logs for [required]

Options:
  -n, --lines INTEGER   Number of log lines to show [default: 100]
  -f, --follow          Follow logs in real-time
  --help                Show this message and exit

Examples:
  lm-start logs Llama-2-7b-chat-hf
  lm-start logs Llama-2-7b-chat-hf --lines 50
  lm-start logs Llama-2-7b-chat-hf --follow
```

### `doctor` - System Diagnostics

```bash
lm-start doctor [OPTIONS]

Run system diagnostics to verify setup.

Checks:
  - OpenCode installation
  - Hardware detection (GPU, CPU, RAM, CUDA)
  - Credentials configuration
  - Python dependencies
  - NVIDIA driver availability

Options:
  --help    Show this message and exit
```

Example output:
```
╭─────────────────────────────────────────╮
│ 🔍 lm-start Doctor - System Diagnostics │
╰─────────────────────────────────────────╯

1. OpenCode Installation
  ✗ OpenCode not found
  Hint: Run lm-start init

2. Hardware Detection
  ✓ CPU Cores: 24
  ✓ RAM: 62.57 GB
  ✓ GPUs: 1
  ✓ GPU Name: NVIDIA GeForce RTX 4090
  ✓ CUDA: Available

3. Credentials Configuration
  ✓ Credentials file found

4. Python Dependencies
  ✓ All dependencies installed

5. NVIDIA Driver
  ✓ nvidia-smi available
```

---

## ⚙️ Configuration

### Directory Structure

```
~/.lm-start/
├── config/
│   ├── credentials.yaml    # Encrypted credentials
│   └── system.yaml         # Hardware profile
└── opencode/
    └── bin/
        └── opencode        # OpenCode binary
```

### credentials.yaml

Stores environment variables securely:

```yaml
version: '1.0'
environment_variables:
  HF_TOKEN: "hf_your_token_here"
  WANDB_API_KEY: "your_key_here"
```

⚠️ **Security Note**: While values are stored in YAML, they are displayed masked in CLI output. Keep this file secure with appropriate file permissions.

### system.yaml

Auto-generated hardware profile:

```yaml
version: '1.0'
hardware:
  gpu_count: 1
  gpu_memory_gb: 24.0
  gpu_name: "NVIDIA GeForce RTX 4090"
  cpu_cores: 24
  ram_gb: 62.6
  python_version: "3.10.12"
  cuda_available: true
  cuda_version: "12.8"
```

---

## 🔧 Troubleshooting

### OpenCode Not Found

**Problem**: Doctor shows "OpenCode not found"

**Solution**:
```bash
lm-start init
```

### GPU Not Detected

**Problem**: Hardware detection shows 0 GPUs

**Solutions**:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Restart system if drivers were recently updated

### Model Download Fails

**Problem**: Setup fails during download phase

**Solutions**:
1. Add HuggingFace token: `lm-start env add HF_TOKEN "your_token"`
2. Check disk space: Ensure sufficient space for model weights
3. Verify network connectivity
4. For gated models, ensure you've accepted terms on HuggingFace

### Setup Interrupted

**Problem**: Setup stopped mid-way

**Solution**:
```bash
lm-start setup <model> --resume
```

### vLLM Startup Failed

**Problem**: Smoke test fails

**Solutions**:
1. Check GPU memory: Ensure model fits in VRAM
2. Try different profile: `lm-start setup <model> --profile speed`
3. Check logs: `lm-start logs <model>`

### Permission Denied

**Problem**: Cannot write to ~/.lm-start/

**Solution**:
```bash
mkdir -p ~/.lm-start
chmod 755 ~/.lm-start
```

---

## 🔒 Security

### Credential Storage

- Credentials are stored in plain YAML at `~/.lm-start/config/credentials.yaml`
- Values are **masked** when displayed via `lm-start env list`
- File permissions should be set to user-only read/write
- **Never commit** credentials.yaml to version control

### Best Practices

1. **Rotate tokens regularly** - Especially HF_TOKEN for gated models
2. **Use environment variables** for CI/CD pipelines instead of stored credentials
3. **Backup credentials.yaml** securely if needed
4. **Review credentials periodically** with `lm-start env list`

---

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- All tests pass
- Documentation is updated

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM inference
- [OpenCode](https://opencode.ai) - AI-powered code assistance
- [PM2](https://pm2.keymetrics.io/) - Process management
- [HuggingFace](https://huggingface.co/) - Model hub

---

## 📞 Support

For issues and feature requests, please check:
1. `lm-start doctor` output
2. Logs with `lm-start logs <model>`
3. This README troubleshooting section

---

**Happy deploying! 🚀**
