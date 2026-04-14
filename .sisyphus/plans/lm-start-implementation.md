# lm-start Implementation Plan

## TL;DR

**Objective**: Build `lm-start` - a Python CLI tool that automates vLLM model deployment with integrated OpenCode agent support.

**Core Value**: One-command setup that installs OpenCode v1.4.0, detects hardware, manages credentials, and deploys vLLM models.

**Deliverables**:
- Python pip package `lm-start`
- CLI with commands: init, env, setup, status, logs, doctor
- Auto-hardware detection
- Credential management
- OpenCode v1.4.0 integration

**Estimated Effort**: Medium (2-3 days)
**Parallel Execution**: YES - 3 waves

---

## Context

### Original Request
User wants to transform their existing shell-based vLLM deployment system (`setup_model.sh`) into a proper Python pip package with:
1. Automatic OpenCode v1.4.0 installation
2. Hardware auto-detection
3. Centralized credential management
4. Simplified CLI interface

### Architecture Decisions
- **Package Type**: Python pip installable
- **OpenCode**: Use official install script with VERSION=1.4.0
- **Config Location**: `~/.lm-start/`
- **Credentials**: Single YAML file with env var export
- **Hardware Detection**: Auto-generate based on system inspection

---

## Work Objectives

### Core Objective
Transform the existing shell-based vLLM deployment system into a professional Python CLI tool with automated setup, credential management, and OpenCode integration.

### Concrete Deliverables
1. `lm-start` pip package with proper setup.py/pyproject.toml
2. CLI entry point with Click/Typer framework
3. Hardware detection module (GPU, CPU, RAM, CUDA)
4. Credential management system
5. OpenCode installer integration
6. Model deployment wrapper (leverages existing logic)
7. Status/monitoring commands

### Definition of Done
- [ ] `pip install lm-start` works correctly
- [ ] `lm-start init` installs OpenCode v1.4.0
- [ ] `lm-start env add KEY value` persists and exports credentials
- [ ] `lm-start setup <model>` deploys vLLM models
- [ ] `lm-start doctor` verifies system health
- [ ] All commands tested and documented

### Must Have
- OpenCode v1.4.0 installation via official script
- Hardware auto-detection (GPU count, VRAM, CPU, RAM)
- Credential storage in ~/.lm-start/config/credentials.yaml
- Environment variable export for OpenCode/vLLM
- Model deployment using existing shell script logic

### Must NOT Have (Guardrails)
- No GUI/web interface (keep it CLI-only)
- No Docker containerization (out of scope)
- No remote/SSH deployment (local only)
- No multi-user support (single user)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: Partial (existing shell scripts)
- **Automated tests**: YES (tests-after implementation)
- **Framework**: pytest + unittest.mock for subprocess calls

### QA Policy
Every task includes agent-executed QA scenarios:
- **CLI commands**: Run command, verify output, check exit code
- **File operations**: Verify files created with correct content
- **Subprocess calls**: Mock or verify actual execution
- **Integration**: End-to-end workflow testing

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - Independent):
├── Task 1: Create pip package structure (setup.py, pyproject.toml)
├── Task 2: Implement CLI framework (Click/Typer, entry points)
├── Task 3: Build hardware detection module (nvidia-smi, lscpu, meminfo)
└── Task 4: Create configuration management (~/.lm-start/ structure)

Wave 2 (Core Features - After Wave 1):
├── Task 5: Implement credential management (env add/list/remove)
├── Task 6: Build OpenCode installer integration (v1.4.0)
├── Task 7: Create init command (orchestrates setup)
└── Task 8: Implement doctor command (system health check)

Wave 3 (Model Deployment - After Wave 2):
├── Task 9: Port existing setup_model.sh logic to Python
├── Task 10: Implement setup command (model deployment)
├── Task 11: Add status/logs commands (PM2 integration)
└── Task 12: Create documentation and README

Wave FINAL (Quality Assurance):
├── Task F1: Plan compliance audit (verify all features implemented)
├── Task F2: Code quality review (linting, type hints)
├── Task F3: Integration testing (full workflow)
└── Task F4: Documentation review
```

### Dependency Matrix

| Task | Dependencies | Blocks |
|------|--------------|--------|
| 1 | None | 2, 3, 4 |
| 2 | 1 | 5, 6, 7, 8 |
| 3 | 1 | 5, 6 |
| 4 | 1 | 5, 6, 7 |
| 5 | 2, 4 | 7, 10 |
| 6 | 2, 3 | 7 |
| 7 | 5, 6, 4 | 10 |
| 8 | 2, 3, 4 | - |
| 9 | 7 | 10 |
| 10 | 5, 7, 9 | 11 |
| 11 | 10 | 12 |
| 12 | 11 | F1-F4 |

---

## TODOs

- [x] 1. Create Python Package Structure

  **What to do**:
  - Create directory structure: `lm_start/`, `tests/`, `docs/`
  - Create `setup.py` or `pyproject.toml` with proper metadata
  - Define entry point: `lm-start = lm_start.cli:main`
  - Add dependencies: click/typer, pyyaml, psutil, rich
  - Create `__init__.py` with version info
  
  **Must NOT do**:
  - Don't add unnecessary dependencies
  - Don't include shell scripts in package (they'll be called)
  
  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []
  - Reason: Straightforward package boilerplate
  
  **Parallelization**:
  - Can Run In Parallel: YES
  - Parallel Group: Wave 1
  - Blocks: Task 2, 3, 4
  - Blocked By: None
  
  **Acceptance Criteria**:
  - [ ] `pip install -e .` succeeds
  - [ ] `lm-start --version` shows version
  - [ ] Package structure follows Python standards
  
  **QA Scenarios**:
  ```
  Scenario: Package installs correctly
    Tool: Bash
    Steps:
      1. cd /home/siraj/Documents/lm_start && pip install -e .
      2. lm-start --version
    Expected Result: Exit code 0, version displayed
    Evidence: .sisyphus/evidence/task-1-package-install.txt
  ```
  
  **Commit**: YES
  - Message: "feat: Initial package structure"
  - Files: setup.py, pyproject.toml, lm_start/__init__.py

- [x] 2. Implement CLI Framework

  **What to do**:
  - Choose Click or Typer (Typer recommended for modern Python)
  - Create main CLI group with subcommands
  - Implement --version flag
  - Add colorful output with Rich library
  - Create placeholder commands: init, env, setup, status, logs, doctor
  - Add help text for all commands
  
  **Must NOT do**:
  - Don't implement command logic yet (placeholders only)
  - Don't use argparse (too low-level)
  
  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []
  - Reason: CLI scaffolding is straightforward
  
  **Parallelization**:
  - Can Run In Parallel: YES (after Task 1)
  - Parallel Group: Wave 1
  - Blocks: Tasks 5, 6, 7, 8
  - Blocked By: Task 1
  
  **Acceptance Criteria**:
  - [ ] `lm-start --help` shows all commands
  - [ ] Each subcommand has --help
  - [ ] Output uses colors/formatting
  
  **QA Scenarios**:
  ```
  Scenario: CLI help works
    Tool: Bash
    Steps:
      1. lm-start --help
      2. lm-start init --help
    Expected Result: Help text displayed, exit code 0
    Evidence: .sisyphus/evidence/task-2-cli-help.txt
  ```
  
  **Commit**: YES
  - Message: "feat: CLI framework with Typer"
  - Files: lm_start/cli.py, lm_start/commands/*.py

- [x] 3. Build Hardware Detection Module

  **What to do**:
  - Detect GPU: Parse `nvidia-smi --query-gpu=name,memory.total,count`
  - Detect CPU: Parse `lscpu` or `/proc/cpuinfo`
  - Detect RAM: Parse `/proc/meminfo` or `psutil.virtual_memory()`
  - Detect CUDA: Check `nvcc --version` or CUDA_HOME
  - Detect Python version: `sys.version_info`
  - Create `SystemInfo` dataclass to hold all data
  - Handle cases where nvidia-smi is not available
  
  **Must NOT do**:
  - Don't require GPU detection to succeed (handle CPU-only systems)
  - Don't use external Python GPU libraries (keep it lightweight)
  
  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []
  - Reason: System-level operations, subprocess parsing
  
  **Parallelization**:
  - Can Run In Parallel: YES (after Task 1)
  - Parallel Group: Wave 1
  - Blocks: Tasks 5, 6, 8
  - Blocked By: Task 1
  
  **Acceptance Criteria**:
  - [ ] Detects GPU count and memory correctly
  - [ ] Detects CPU cores correctly
  - [ ] Detects RAM correctly
  - [ ] Handles missing nvidia-smi gracefully
  
  **QA Scenarios**:
  ```
  Scenario: Hardware detection works
    Tool: Bash (Python REPL)
    Steps:
      1. python -c "from lm_start.system import detect_hardware; print(detect_hardware())"
    Expected Result: Dictionary with gpu_count, gpu_memory, cpu_cores, ram_gb
    Evidence: .sisyphus/evidence/task-3-hardware-detect.txt
  
  Scenario: Handles missing GPU
    Tool: Bash
    Steps:
      1. SSH into CPU-only machine
      2. Run hardware detection
    Expected Result: gpu_count=0, no crash
    Evidence: .sisyphus/evidence/task-3-no-gpu.txt
  ```
  
  **Commit**: YES
  - Message: "feat: Hardware detection module"
  - Files: lm_start/system.py, tests/test_system.py

- [x] 4. Create Configuration Management

  **What to do**:
  - Create `~/.lm-start/` directory structure
  - Create `~/.lm-start/config/` subdirectory
  - Define config file formats (YAML for human-readable)
  - Implement config loader/saver with validation
  - Create default configs for system.yaml
  - Handle config migrations (version field)
  
  **Must NOT do**:
  - Don't use JSON for user-editable configs (YAML is friendlier)
  - Don't create configs in current directory (always ~/.lm-start/)
  
  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []
  - Reason: File I/O and YAML handling
  
  **Parallelization**:
  - Can Run In Parallel: YES (after Task 1)
  - Parallel Group: Wave 1
  - Blocks: Tasks 5, 6, 7
  - Blocked By: Task 1
  
  **Acceptance Criteria**:
  - [ ] Creates directory structure on first run
  - [ ] Loads/saves YAML configs correctly
  - [ ] Validates config schema
  
  **QA Scenarios**:
  ```
  Scenario: Config directory creation
    Tool: Bash
    Steps:
      1. rm -rf ~/.lm-start
      2. python -c "from lm_start.config import ensure_config_dir; ensure_config_dir()"
      3. ls -la ~/.lm-start/
    Expected Result: Directory created with config/ subdirectory
    Evidence: .sisyphus/evidence/task-4-config-dir.txt
  ```
  
  **Commit**: YES
  - Message: "feat: Configuration management"
  - Files: lm_start/config.py, lm_start/constants.py

- [x] 5. Implement Credential Management

  **What to do**:
  - Create `~/.lm-start/config/credentials.yaml` format
  - Implement `lm-start env add KEY value` command
  - Implement `lm-start env add KEY --from-env` (read from environment)
  - Implement `lm-start env list` command (masked values)
  - Implement `lm-start env remove KEY` command
  - Export credentials to shell when running other commands
  - Option to persist to ~/.bashrc or ~/.zshrc
  
  **Credentials Format**:
  ```yaml
  version: "1.0"
  environment_variables:
    HF_TOKEN: "hf_xxxxx"
    WANDB_API_KEY: "xxxx"
  shell_rc: "~/.bashrc"
  ```
  
  **Must NOT do**:
  - Don't store credentials in plain text without warning
  - Don't require all credentials (make optional)
  
  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []
  - Reason: File I/O, command implementation
  
  **Parallelization**:
  - Can Run In Parallel: YES (after Tasks 2, 4)
  - Parallel Group: Wave 2
  - Blocks: Tasks 7, 10
  - Blocked By: Tasks 2, 4
  
  **Acceptance Criteria**:
  - [ ] `env add` creates credentials.yaml
  - [ ] Values are masked in `env list`
  - [ ] Credentials exported to environment
  - [ ] Optional: Added to shell RC file
  
  **QA Scenarios**:
  ```
  Scenario: Add and list credentials
    Tool: Bash
    Steps:
      1. lm-start env add HF_TOKEN "hf_test123"
      2. lm-start env list
      3. cat ~/.lm-start/config/credentials.yaml
    Expected Result: 
      - Command 1: Success message
      - Command 2: Shows HF_TOKEN=hf_t******** (masked)
      - Command 3: Contains full token
    Evidence: .sisyphus/evidence/task-5-credentials.txt
  ```
  
  **Commit**: YES
  - Message: "feat: Credential management commands"
  - Files: lm_start/commands/env.py, tests/test_env.py

- [x] 6. Build OpenCode Installer Integration

  **What to do**:
  - Download OpenCode v1.4.0 using official install script
  - Set `VERSION=1.4.0` and `OPENCODE_INSTALL_DIR=$HOME/.lm-start/opencode`
  - Verify installation succeeded
  - Create symlink or wrapper for easy access
  - Handle existing installations (skip or reinstall)
  - Verify binary works: `opencode --version` == 1.4.0
  
  **Must NOT do**:
  - Don't modify user's PATH permanently (use shim/wrapper)
  - Don't install latest version (must be 1.4.0)
  
  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []
  - Reason: Subprocess execution, file operations
  
  **Parallelization**:
  - Can Run In Parallel: YES (after Tasks 2, 3)
  - Parallel Group: Wave 2
  - Blocks: Task 7
  - Blocked By: Tasks 2, 3
  
  **Acceptance Criteria**:
  - [ ] Installs OpenCode to ~/.lm-start/opencode/
  - [ ] Version is exactly 1.4.0
  - [ ] Binary is executable
  
  **QA Scenarios**:
  ```
  Scenario: OpenCode installation
    Tool: Bash
    Steps:
      1. rm -rf ~/.lm-start/opencode
      2. python -c "from lm_start.opencode import install; install('1.4.0')"
      3. ~/.lm-start/opencode/bin/opencode --version
    Expected Result: Version is 1.4.0
    Evidence: .sisyphus/evidence/task-6-opencode-install.txt
  ```
  
  **Commit**: YES
  - Message: "feat: OpenCode v1.4.0 installer"
  - Files: lm_start/opencode.py

- [x] 7. Create Init Command

  **What to do**:
  - Orchestrate full setup process
  - Check if OpenCode installed (install if not)
  - Run hardware detection
  - Generate system.yaml config
  - Prompt for credentials (optional)
  - Display summary of detected system
  - Set up ~/.lm-start/ directory structure
  
  **Init Flow**:
  1. Check/create ~/.lm-start/ directory
  2. Install OpenCode v1.4.0 if not present
  3. Detect hardware (GPUs, CPU, RAM)
  4. Generate system.yaml
  5. Prompt for HF_TOKEN (optional)
  6. Display summary
  
  **Must NOT do**:
  - Don't fail if hardware detection has issues (warn instead)
  - Don't require credentials (make optional)
  
  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []
  - Reason: Orchestration, user interaction
  
  **Parallelization**:
  - Can Run In Parallel: NO (depends on 5, 6, 4)
  - Parallel Group: Wave 2 (sequential)
  - Blocks: Task 10
  - Blocked By: Tasks 5, 6, 4
  
  **Acceptance Criteria**:
  - [ ] Completes full setup流程
  - [ ] Displays system summary
  - [ ] Creates all necessary files
  
  **QA Scenarios**:
  ```
  Scenario: Fresh init
    Tool: Bash
    Steps:
      1. rm -rf ~/.lm-start
      2. lm-start init (provide HF_TOKEN when prompted)
      3. ls -la ~/.lm-start/
      4. cat ~/.lm-start/config/system.yaml
    Expected Result: All directories created, OpenCode installed, system.yaml populated
    Evidence: .sisyphus/evidence/task-7-init.txt
  ```
  
  **Commit**: YES
  - Message: "feat: Init command orchestration"
  - Files: lm_start/commands/init.py

- [x] 8. Implement Doctor Command

  **What to do**:
  - Verify OpenCode installation (version 1.4.0)
  - Verify hardware detection results
  - Check credential file exists and is readable
  - Verify Python dependencies installed
  - Check disk space for models
  - Test nvidia-smi availability
  - Report any issues with suggestions
  
  **Must NOT do**:
  - Don't require all checks to pass (informational)
  - Don't modify anything (read-only diagnostic)
  
  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []
  - Reason: Diagnostic checks, reporting
  
  **Parallelization**:
  - Can Run In Parallel: YES (after Tasks 2, 3, 4)
  - Parallel Group: Wave 2
  - Blocks: None
  - Blocked By: Tasks 2, 3, 4
  
  **Acceptance Criteria**:
  - [ ] Reports OpenCode status
  - [ ] Reports hardware status
  - [ ] Reports credential status
  - [ ] Suggests fixes for issues
  
  **QA Scenarios**:
  ```
  Scenario: Doctor on healthy system
    Tool: Bash
    Steps:
      1. lm-start doctor
    Expected Result: All checks pass, green checkmarks
    Evidence: .sisyphus/evidence/task-8-doctor-healthy.txt
  
  Scenario: Doctor on incomplete system
    Tool: Bash
    Steps:
      1. rm -rf ~/.lm-start/opencode
      2. lm-start doctor
    Expected Result: Reports OpenCode missing, suggests fix
    Evidence: .sisyphus/evidence/task-8-doctor-incomplete.txt
  ```
  
  **Commit**: YES
  - Message: "feat: Doctor diagnostic command"
  - Files: lm_start/commands/doctor.py

- [x] 9. Port Existing Setup Logic

  **What to do**:
  - Study existing `setup_model.sh` phases
  - Port phase logic to Python functions
  - Maintain state management (JSON persistence)
  - Call external scripts when needed (Python wrappers)
  - Preserve agentic recovery capability
  - Handle resume logic
  
  **Phases to Port**:
  1. init - Directory setup
  2. fetch_info - Get model metadata from HF
  3. download - Download model weights
  4. venv - Create Python virtualenv with vLLM
  5. smoke_test - Test vLLM startup
  6. extract_vllm_config - Get runtime config
  7. generate_config - Create PM2 config
  8. optimize - Run optimization agents
  9. finalize - Create README and scripts
  
  **Must NOT do**:
  - Don't rewrite everything in Python (call existing scripts when complex)
  - Don't break existing state file format
  
  **Recommended Agent Profile**:
  - Category: `deep`
  - Skills: []
  - Reason: Complex logic migration, state management
  
  **Parallelization**:
  - Can Run In Parallel: NO (depends on 7)
  - Parallel Group: Wave 3 (sequential)
  - Blocks: Task 10
  - Blocked By: Task 7
  
  **Acceptance Criteria**:
  - [ ] All 9 phases ported
  - [ ] State management preserved
  - [ ] Can resume interrupted setups
  
  **QA Scenarios**:
  ```
  Scenario: Phase execution
    Tool: Bash
    Steps:
      1. Run individual phases via Python API
      2. Check state file updates
    Expected Result: Phases execute, state persisted
    Evidence: .sisyphus/evidence/task-9-phases.txt
  ```
  
  **Commit**: YES
  - Message: "feat: Port setup phases to Python"
  - Files: lm_start/phases/*.py, lm_start/state.py

- [x] 10. Implement Setup Command

  **What to do**:
  - Create `lm-start setup <model_name>` command
  - Accept options: --profile, --optimize, --resume
  - Execute all 9 phases
  - Show progress with spinners/progress bars
  - Handle errors gracefully
  - Display final summary
  
  **Command Signature**:
  ```bash
  lm-start setup Qwen/Qwen2.5-32B [--profile speed|balanced|quality] [--optimize] [--resume]
  ```
  
  **Must NOT do**:
  - Don't run if init hasn't been done (check and warn)
  - Don't proceed if hardware insufficient (warn)
  
  **Recommended Agent Profile**:
  - Category: `deep`
  - Skills: []
  - Reason: Complex orchestration, error handling
  
  **Parallelization**:
  - Can Run In Parallel: NO (depends on 5, 7, 9)
  - Parallel Group: Wave 3 (sequential)
  - Blocks: Task 11
  - Blocked By: Tasks 5, 7, 9
  
  **Acceptance Criteria**:
  - [ ] Command accepts model name
  - [ ] All options work correctly
  - [ ] Progress displayed to user
  - [ ] Errors handled gracefully
  
  **QA Scenarios**:
  ```
  Scenario: Setup with small model
    Tool: Bash
    Steps:
      1. lm-start setup TinyLlama/TinyLlama-1.1B-Chat-v0.4
    Expected Result: All phases complete, model deployed
    Evidence: .sisyphus/evidence/task-10-setup.txt
  ```
  
  **Commit**: YES
  - Message: "feat: Setup command for model deployment"
  - Files: lm_start/commands/setup.py

- [x] 11. Implement Status and Logs Commands

  **What to do**:
  - `lm-start status` - Show all deployed models
  - `lm-start status <model>` - Show specific model status
  - `lm-start logs <model>` - Show logs (latest N lines)
  - `lm-start logs <model> -f` - Follow logs in real-time
  - Integrate with PM2 (parse `pm2 status`, `pm2 logs`)
  - Display GPU usage, memory, etc.
  
  **Must NOT do**:
  - Don't require PM2 if not installed (graceful degradation)
  - Don't fail if model not found (informative error)
  
  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []
  - Reason: Subprocess parsing, output formatting
  
  **Parallelization**:
  - Can Run In Parallel: NO (depends on 10)
  - Parallel Group: Wave 3 (sequential)
  - Blocks: Task 12
  - Blocked By: Task 10
  
  **Acceptance Criteria**:
  - [ ] Status shows model state
  - [ ] Logs displays correctly
  - [ ] Follow mode works
  
  **QA Scenarios**:
  ```
  Scenario: Check status
    Tool: Bash
    Steps:
      1. lm-start status
      2. lm-start logs <deployed_model>
    Expected Result: Shows deployed models and their logs
    Evidence: .sisyphus/evidence/task-11-status-logs.txt
  ```
  
  **Commit**: YES
  - Message: "feat: Status and logs commands"
  - Files: lm_start/commands/status.py, lm_start/commands/logs.py

- [x] 12. Create Documentation

  **What to do**:
  - Write comprehensive README.md
  - Document all commands with examples
  - Create installation guide
  - Add troubleshooting section
  - Document configuration options
  - Add architecture diagram/description
  
  **Must NOT do**:
  - Don't duplicate existing vLLM docs
  - Don't forget to document credentials security
  
  **Recommended Agent Profile**:
  - Category: `writing`
  - Skills: []
  - Reason: Documentation, clear explanations
  
  **Parallelization**:
  - Can Run In Parallel: NO (depends on 11)
  - Parallel Group: Wave 3 (sequential)
  - Blocks: F1-F4
  - Blocked By: Task 11
  
  **Acceptance Criteria**:
  - [ ] README complete with all commands
  - [ ] Installation instructions clear
  - [ ] Troubleshooting section added
  
  **QA Scenarios**:
  ```
  Scenario: Documentation review
    Tool: Manual review
    Steps:
      1. Read README.md
      2. Verify all commands documented
    Expected Result: Complete documentation
    Evidence: .sisyphus/evidence/task-12-docs.png
  ```
  
  **Commit**: YES
  - Message: "docs: Complete README and documentation"
  - Files: README.md, docs/

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** - Verify all 12 tasks completed, all features implemented, documentation complete
- [x] F2. **Code Quality Review** - Run black/flake8, add type hints, ensure consistent style
- [x] F3. **Integration Testing** - Full end-to-end test: init → env add → setup → status → logs
- [x] F4. **Documentation Review** - Verify README accuracy, check all examples work

---

## Commit Strategy

- Wave 1 commits: tasks 1-4 (foundational)
- Wave 2 commits: tasks 5-8 (core features)
- Wave 3 commits: tasks 9-12 (deployment + docs)
- Final commits: F1-F4 (polish)

---

## Success Criteria

### Verification Commands
```bash
# Installation
pip install lm-start
lm-start --version

# Initialization
lm-start init

# Credentials
lm-start env add HF_TOKEN "test_token"
lm-start env list

# System check
lm-start doctor

# Model deployment (small model for testing)
lm-start setup TinyLlama/TinyLlama-1.1B-Chat-v0.4

# Monitoring
lm-start status
lm-start logs TinyLlama-1.1B-Chat-v0.4
```

### Final Checklist
- [ ] All commands work correctly
- [ ] OpenCode v1.4.0 installed correctly
- [ ] Hardware detection accurate
- [ ] Credentials persist and export
- [ ] Model deployment succeeds
- [ ] Documentation complete
