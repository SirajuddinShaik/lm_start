# lm-start Project Summary

## Overview

**lm-start** is an intelligent automation framework that transforms the complex, error-prone process of deploying Large Language Models (LLMs) locally into a single, reliable command. It handles everything from hardware detection to production-ready vLLM deployment with automatic optimization.

### What It Does

```bash
# One command to deploy any model
lm-start setup meta-llama/Llama-2-7b-chat-hf

# That's it. lm-start will:
# 1. Detect your GPUs (H100, A100, RTX 4090, etc.)
# 2. Download the model from HuggingFace
# 3. Install vLLM with correct CUDA version
# 4. Run smoke tests to verify it works
# 5. AI agents automatically optimize configuration
# 6. Deploy with PM2 for production
```

### The Problem It Solves

**Without lm-start:**
- Manual vLLM configuration (100+ flags to tune)
- Trial-and-error to find working batch sizes
- CUDA version mismatches
- Out-of-memory crashes at high context
- No systematic benchmarking
- Hours wasted per model deployment

**With lm-start:**
- Fully automated 9-phase pipeline
- AI agents test configurations intelligently
- Hardware-optimized settings
- Comprehensive benchmarking (32K/64K context + stress tests)
- Resume interrupted deployments
- 10-30 minutes to production-ready deployment

### Core Capabilities

| Feature | What It Means |
|---------|--------------|
| **Hardware Auto-Detection** | Automatically detects GPUs, CUDA version, RAM, and computes optimal settings |
| **9-Phase Pipeline** | Structured deployment: Init → Fetch → Download → Venv → Smoke Test → Extract Config → Generate → Optimize → Finalize |
| **AI-Driven Optimization** | Agents run experiments (`run_1_1`, `run_1_2`, etc.) to find best throughput/memory tradeoff |
| **Tree-Based Experiments** | Hierarchical experiment IDs track optimization lineage (e.g., `run_1_1` → `run_1_2` is optimizing the first config) |
| **Dual-Agent System** | Planner designs experiments (read-only), Summarizer documents results (write-only) to Insights.md |
| **Multi-Benchmark Suite** | Tests at 32K/64K context + stress test at 75% capacity to ensure stability |
| **Production Deployment** | Automatically generates PM2 configs for process management |
| **Resume Capability** | Interrupted deployment? `lm-start setup <model> --resume` continues from last phase |

### Key Innovations

1. **Intelligent Configuration Search**: Instead of grid search, AI agents use tree-based optimization with convergence detection (stop when no improvement)

2. **Separation of Concerns**: Planner designs configs (never touches Insights.md), Summarizer documents results (owns Insights.md) - eliminates file corruption

3. **Config Profiles**: Track optimization goals (`optimized_for: throughput|balanced|quality`) to compare speed vs accuracy tradeoffs

4. **Automatic vLLM Matching**: Installs correct vLLM wheel (cu130, etc.) matching your system CUDA

5. **Comprehensive Validation**: Not just "it starts" - tests 32K/64K context AND stress tests at 75% capacity

### Ideal Use Cases

- **ML Engineers**: Deploy models for internal APIs
- **Researchers**: Test models on private datasets locally
- **Startups**: Production LLM serving without managed services
- **Hobbyists**: Run large models on consumer GPUs (with auto-optimization)

### Supported Hardware

- NVIDIA GPUs: H100, H200, A100, A6000, RTX 4090/3090/2080Ti, etc.
- Multi-GPU setups (automatic tensor parallelism)
- Consumer to enterprise-grade deployments

---

## CLI Commands

### Core Commands

| Command | Description |
|---------|-------------|
| `lm-start init` | Initialize lm-start configuration directory (~/.lm-start/) |
| `lm-start doctor` | Run system diagnostics (hardware, deps, NVIDIA drivers) |
| `lm-start setup <model>` | Setup and deploy a model (9-phase pipeline) |
| `lm-start status [model]` | Show deployment status of models |
| `lm-start logs <model>` | Show deployment logs for a model |

### Environment Management

| Command | Description |
|---------|-------------|
| `lm-start env add <key> <value>` | Add credential/environment variable |
| `lm-start env list` | List stored credentials (masked) |
| `lm-start env remove <key>` | Remove a credential |

### Session Management

| Command | Description |
|---------|-------------|
| `lm-start session show <id>` | Show session details |
| `lm-start session list` | List all sessions |
| `lm-start session open <id>` | Open session in default editor |
| `lm-start session last` | Show last session |
| `lm-start session delete <id>` | Delete a session |
| `lm-start session clear` | Clear all sessions |
| `lm-start session sync` | Sync sessions across workspaces |

### Other Commands

| Command | Description |
|---------|-------------|
| `lm-start version` | Show version information |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    lm-start Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CLI Layer (commands/)        Core Layer (core/)            │
│  ├── init.py                  ├── config.py                 │
│  ├── setup.py                 ├── hardware.py               │
│  ├── env.py                   └── state.py                  │
│  ├── doctor.py                                              │
│  ├── status.py                                              │
│  ├── logs.py                                                │
│  └── session.py                                             │
│                                                             │
│  Phase Layer (phases/)        Agent Layer (agents/)         │
│  ├── Phase implementations    ├── ExperimentAgent           │
│  └── State management         ├── PhaseAgent                │
│                               └── BaseAgent                 │
│                                                             │
│  Scripts (scripts/)           Utils (utils/)                │
│  ├── Model operations         ├── BenchmarkRunner           │
│  ├── Config generation        ├── CapacityCalculator        │
│  └── Optimization             └── FlagSelector              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9-Phase Deployment Pipeline

### Phase 1: INIT
**Purpose**: Initialize model directory structure and state

**Actions**:
- Create model directory (`~/.lm-start/models/<model>/`)
- Initialize `.runs/` subdirectory for experiment tracking with tree-based IDs
- Create `setup_state.json` for resumable deployments
- Set up logging infrastructure

**Key Files**:
- `lm_start/phases/__init__.py:144` - `phase_init()`

**Outputs**:
- Model directory structure
- Initial state file

---

### Phase 2: FETCH_INFO
**Purpose**: Retrieve model metadata from HuggingFace

**Actions**:
- Query HuggingFace Hub for model information
- Extract model architecture, parameters, context length
- Determine quantization requirements
- Check model compatibility with vLLM

**Key Files**:
- `lm_start/phases/__init__.py:224` - `phase_fetch_info()`
- `lm_start/scripts/fetch_model_info.py`

**Outputs**:
- `model_info.json` - Model metadata
- `device_config.json` - Hardware profile

---

### Phase 3: DOWNLOAD
**Purpose**: Download model weights and tokenizer

**Actions**:
- Download model files from HuggingFace
- Handle gated models (requires HF_TOKEN)
- Resume interrupted downloads
- Verify checksums

**Key Files**:
- `lm_start/phases/__init__.py:272` - `phase_download()`
- `lm_start/scripts/download_model.py`

**Outputs**:
- Model weights in `~/.cache/huggingface/`
- Tokenizer files

---

### Phase 4: VENV
**Purpose**: Create isolated Python environment with vLLM

**Actions**:
- Create virtual environment per model
- Install vLLM with CUDA wheel matching system (cu130 for stability)
- Install model-specific dependencies
- Configure environment variables

**Key Files**:
- `lm_start/phases/__init__.py:320` - `phase_venv()`
- `lm_start/scripts/create_venv.sh`

**Outputs**:
- `.venv/` directory with vLLM installed
- Activation scripts

---

### Phase 5: SMOKE_TEST
**Purpose**: Validate vLLM can start with basic configuration

**Actions**:
- Start vLLM server with conservative config
- Perform health check
- Test basic inference
- Verify GPU utilization

**Key Files**:
- `lm_start/phases/__init__.py:390` - `phase_smoke_test()`

**Outputs**:
- Smoke test results
- Server logs

---

### Phase 6: EXTRACT_VLLM_CONFIG
**Purpose**: Extract vLLM's runtime configuration

**Actions**:
- Start vLLM and capture `--help-all` output
- Parse available flags and defaults
- Detect model-specific optimizations
- Extract memory estimates

**Key Files**:
- `lm_start/phases/__init__.py:670` - `phase_extract_vllm_config()`
- `lm_start/scripts/extract_vllm_config.py`
- `lm_start/scripts/extract_vllm_flags.py`

**Outputs**:
- `vllm_config.json` - Runtime configuration
- Flag compatibility matrix

---

### Phase 7: GENERATE_CONFIG
**Purpose**: Generate optimized vLLM configuration

**Actions**:
- Calculate theoretical capacity from model + hardware
- Generate initial vLLM flags with `optimized_for` field (throughput/balanced/quality)
- Create PM2 ecosystem configuration
- Set up monitoring endpoints

**Key Files**:
- `lm_start/phases/__init__.py:857` - `phase_generate_config()`
- `lm_start/scripts/generate_pm2_config.py`

**Outputs**:
- `ecosystem.config.js` - PM2 configuration
- `vllm_config.json` - Optimized flags with `optimized_for` field

---

### Phase 8: OPTIMIZE
**Purpose**: Run agent-driven configuration optimization

**Actions**:
- **Experiment Agent** tests multiple configurations
- Generate configs with tree-based naming (`run_1_1`, `run_1_2`, `run_2_1`)
- Start vLLM with each config
- **Run Benchmark Array**:
  - Comprehensive benchmark (32K/64K context)
  - Stress test (75% concurrency at max context)
- Aggregate results and select best config
- **Summarizer** updates Insights.md with tree structure and `Optimized` column
- Iteratively refine until stable

**Key Files**:
- `lm_start/phases/__init__.py:897` - `phase_optimize()`
- `lm_start/agents/experiment_agent.py`
- `lm_start/utils/benchmark_runner.py`
- `lm_start/prompts/opencode_planner.yaml` - Tree-based naming, intelligent planning
- `lm_start/prompts/opencode_summarizer.yaml` - Insights.md ownership, tree structure

**Outputs**:
- Optimized `vllm_config.json`
- `benchmark_<timestamp>.json` - Benchmark results
- `.runs/` directory with experiment history and tree structure
- `Insights.md` - Human-readable summary with best config

**Benchmark Array**:
```python
BENCHMARK_REGISTRY = [
    {
        "name": "comprehensive",
        "method": "run_comprehensive_benchmark",
        "tests": ["32K context", "64K context"]
    },
    {
        "name": "stress_test", 
        "method": "run_stress_test",
        "params": {"target_percent": 75, "seq_headroom": 2048}
    }
]
```

---

### Phase 9: FINALIZE
**Purpose**: Complete deployment and create documentation

**Actions**:
- Create startup/shutdown scripts
- Set up log rotation
- Create management commands

**Key Files**:
- `lm_start/phases/__init__.py:973` - `phase_finalize()`
- `lm_start/scripts/generate_model_sh.py`

**Outputs**:
- `README.md` - Model-specific documentation
- `start.sh` / `stop.sh` - Management scripts
- Deployment complete marker

---

## Prompt Architecture

### Planner Prompt (`prompts/opencode_planner.yaml`)
**Purpose**: Generate experiment configurations

**Key Sections**:
- `experiment_naming` - Tree-based ID format: `run_{tree}_{id}`
  - ROOT: `tree == id` (e.g., `run_1_1`, `run_2_2`)
  - CHILD: `tree < id` (e.g., `run_1_2`, `run_1_3`)
- `file_permissions` - Insights.md READ-ONLY for planner
- `intelligent_planning` - No grid search, convergence criteria
  - Default: 1 experiment per iteration
  - Parallel only for backend comparison (max 3)
  - STOP after 3 consecutive non-improvements
- `parent_linking` - Children learn from parent experiments
- `outputs` - Planner outputs `experiments_queued` JSON only

**Output Format**:
```yaml
experiments_queued: [
  {
    "name": "run_{tree}_{id}",
    "config": {
      "optimized_for": "balanced",  # throughput|balanced|quality
      "max_model_len": ...,
      ...
    }
  }
]
```

### Summarizer Prompt (`prompts/opencode_summarizer.yaml`)
**Purpose**: Document results in Insights.md

**Key Sections**:
- `file_ownership` - "YOU OWN Insights.md" - exclusive write access
- `insights_template` - Tree-based structure with `Optimized` column
  ```markdown
  | ID | Optimized | Config | Result | Key Finding |
  |----|-----------|--------|--------|-------------|
  | 1_1 | balanced | ... | ✅ 850 | ... |
  ```
- `refactoring_rules` - Auto-refactor when >500 lines
- `optimized_for_extraction` - Extract from config or infer from flags
  - throughput: high batch, chunked prefill
  - quality: eager mode, conservative
  - balanced: middle values (default)

**Role Separation**:
- Planner: READ-ONLY from Insights.md
- Summarizer: WRITE-ONLY to Insights.md

---

## Scripts Directory

### Model Operations

| Script | Purpose | Key Functions |
|--------|---------|---------------|
| `fetch_model_info.py` | Query HuggingFace for model metadata | `fetch_model_info_func()` - Downloads model card, extracts architecture info |
| `download_model.py` | Download model weights | `download_model_func()` - Handles resume, gated models, verification |
| `extract_vllm_config.py` | Extract vLLM runtime config | Extracts actual vLLM defaults and flags |
| `extract_vllm_flags.py` | Parse vLLM flags | Generates flag knowledge base |

### Configuration Generation

| Script | Purpose | Key Functions |
|--------|---------|---------------|
| `generate_pm2_config.py` | Create PM2 ecosystem file | Generates process management config |
| `generate_model_sh.py` | Create startup scripts | Shell scripts for model lifecycle |
| `deploy.py` | Deploy model with config | Full deployment orchestration |

### Optimization & Agents

| Script | Purpose | Key Functions |
|--------|---------|---------------|
| `optimizer.py` | Configuration optimizer | Grid search and optimization algorithms |
| `opencode.py` | OpenCode integration | AI assistant for troubleshooting |
| `opencode_phase_agent.py` | Phase-specific AI agent | Recover from phase failures |
| `orchestrator.py` | Experiment orchestration | Manages experiment runs |

### CLI & Interface

| Script | Purpose | Key Functions |
|--------|---------|---------------|
| `cli.py` | Command-line interface | Main entry point for scripts |

---

## Utility Modules

### Benchmarking (`utils/benchmark_runner.py`)

**BenchmarkRunner Class**:
- `run_serve_benchmark()` - Execute vLLM bench serve
- `run_comprehensive_benchmark()` - Test at 32K/64K contexts
- `run_stress_test()` - High concurrency validation at 75% capacity

**Stress Test**:
```python
# Runs at: 75% of max_num_seqs concurrent requests
# Sequence: max_model_len - 2k tokens
# Duration: 30-90 seconds
# Reports: crashed/success, metrics, test_parameters
```

### Capacity Calculation (`utils/capacity_calculator.py`)

**CapacityCalculator Class**:
- Calculate theoretical serving capacity
- Account for model weights, KV cache, activations
- GPU memory budgeting
- Recommend max_num_seqs and batch sizes

### Flag Selection (`utils/vllm_flag_selector.py`)

**VLLMFlagSelector Class**:
- Select optimal vLLM flags based on hardware
- Attention backend selection (FLASHINFER, FLASH_ATTN, etc.)
- Quantization recommendations
- Memory optimization strategies
- Profile-based recommendations (speed/balanced/quality)

---

## Agent Architecture

### Experiment Agent (`agents/experiment_agent.py`)

**Purpose**: Autonomously find optimal vLLM configuration using tree-based experiments

**Algorithm**:
1. Generate initial config from theoretical calculations with `optimized_for` field
2. Start vLLM server
3. Run BENCHMARK_REGISTRY array
4. If all_passed: SUCCESS
5. If failed: Modify config, create child experiment (e.g., `run_1_2`), retry

**Configuration Parameters Tested**:
- `max_num_batched_tokens` - Prefill batch size
- `max_num_seqs` - Concurrent requests
- `gpu_memory_utilization` - VRAM usage
- `attention_backend` - FlashAttention variant
- `enable_chunked_prefill` - Chunking strategy
- `optimized_for` - Optimization goal (throughput/balanced/quality)

### Phase Agent (`agents/phase_agent.py`)

**Purpose**: Recover from phase failures using AI

**Triggered**: When any phase fails
**Actions**:
- Analyze error logs
- Consult OpenCode for solutions
- Apply fixes
- Retry phase

---

## Configuration System

### system.yaml (`~/.lm-start/config/system.yaml`)

Central configuration file controlling all aspects:

```yaml
# Hardware detection
hardware:
  gpu: {count, memory_gb, name}
  cuda: {version, home}

# Experiment settings
experiment:
  max_runs: 20
  benchmarks:
    - name: comprehensive
      enabled: true
    - name: stress_test
      enabled: true
      target_percent: 75
      seq_headroom: 2048

# vLLM defaults
vllm_defaults:
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 8
```

---

## State Management

### State Persistence

**File**: `setup_state.json` in model directory

Tracks:
- Current phase
- Completed phases
- Error history
- Resume capability

**StateManager Class** (`state_manager.py`):
- `start_phase()` - Mark phase as in_progress
- `complete_phase()` - Mark phase completed
- `fail_phase()` - Record failure with error
- `can_resume()` - Check if deployment can resume

---

## Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Hardware   │────▶│   Theoretical│────▶│    Initial   │
│   Detection  │     │   Capacity   │     │    Config    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Optimized  │◀────│   Benchmark  │◀────│  Start vLLM  │
│    Config    │     │     Array    │     │   Server     │
└──────────────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────┐
│  Deploy to   │
│     PM2      │
└──────────────┘
```

---

## Development Commands

```bash
# Development installation
pip install -e .

# Run tests
pytest tests/ -v

# Check specific module
python -m lm_start.doctor

# Manual phase execution
python -m lm_start.scripts.fetch_model_info <model_dir>
python -m lm_start.scripts.extract_vllm_config <model_dir>
```

---

## File Structure Summary

```
lm_start/
├── agents/           # AI agents for optimization
├── commands/         # CLI commands (Typer)
│   ├── init.py
│   ├── doctor.py
│   ├── setup.py
│   ├── status.py
│   ├── logs.py
│   ├── env.py
│   └── session.py
├── core/             # Core functionality
├── phases/           # 9-phase pipeline
├── scripts/          # Sub-scripts
├── utils/            # Utilities
└── prompts/          # AI agent prompts
    ├── opencode_planner.yaml      # Tree-based naming, intelligent planning
    ├── opencode_summarizer.yaml   # Insights.md ownership, tree structure
    ├── agent_optimization.yaml
    └── ...

models/
└── <model_name>/
    ├── .runs/           # Experiment history with tree-based IDs
    │   └── run_X_Y/
    │       ├── config.json
    │       ├── result.json
    │       ├── benchmark.json
    │       └── summary.json
    ├── .llm-context/    # Model context
    ├── Insights.md      # Human-readable summary (tree structure)
    ├── benchmark_*.json # Benchmark results
    ├── ecosystem.config.js
    ├── vllm_config.json
    └── setup_state.json
```

---

## Key Improvements (Recent)

### Tree-Based Experiment Naming
- IDs: `run_{tree}_{id}` where both numbers are sequential
- Examples: `run_1_1`, `run_1_2`, `run_1_3` (tree 1); `run_2_4` (tree 2 root)
- Clear parent-child relationships for optimization tracking

### Insights.md Ownership Model
- **Planner**: READ-ONLY access to Insights.md
- **Summarizer**: WRITE-ONLY access to Insights.md
- Eliminates conflicting edits and messy files
- Auto-refactoring when >500 lines

### Intelligent Planning
- Default: 1 experiment per iteration (focused optimization)
- Parallel only for backend comparison (max 3)
- Convergence criteria: stop after 3 consecutive non-improvements
- No grid-search behavior

### Config Profile Tracking
- `optimized_for` field: throughput | balanced | quality
- Visible in Insights.md tables
- Easy comparison of speed vs quality runs

---

## License

MIT License - See LICENSE file for details

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/ -v`
4. Submit pull request

For questions or issues, check `lm-start doctor` output first.
