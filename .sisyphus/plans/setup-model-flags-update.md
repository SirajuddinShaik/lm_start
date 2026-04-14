# Update setup_model.sh Flags

## TL;DR

Modify `setup_model.sh` to:
1. Make `--optimize` and `--agentic` enabled **by default** (automatic)
2. Add `--deep` flag to trigger deep optimization mode
3. Pass optimization mode to planner via environment/context

User just runs: `./setup_model.sh model/name` (optimize + agentic automatic)
For deep mode: `./setup_model.sh model/name --deep`

## Context

Current behavior (from reading setup_model.sh lines 1008-1069):
- Line 1008: `OPTIMIZE=false` (default)
- Line 1009: `OPTIMIZE_AGENTIC=false` (default)
- Line 1065: `OPTIMIZE=true` when `--optimize` flag passed
- Line 1069: `OPTIMIZE_AGENTIC=true` when `--agentic` flag passed
- Planner spawned via `opencode_agentic_orchestrator.py` line 835: `OPTIMIZE_ARGS="$OPTIMIZE_ARGS --agentic"`

Desired behavior:
- `OPTIMIZE=true` and `OPTIMIZE_AGENTIC=true` by default (automatic)
- `--deep` flag enables thorough backend/CUDA graph research mode
- No changes needed to use optimization - it's automatic

## Work Objectives

### Core Objective
Update setup_model.sh flag handling to enable optimize/agentic by default and add deep mode support.

### Concrete Deliverables
1. Modified `setup_model.sh`:
   - `--optimize` and `--agentic` enabled by default
   - `--deep` flag added for deep mode
2. Update argument parsing logic
3. Pass optimization mode to orchestrator/planner

### Definition of Done
- [ ] --optimize is default behavior
- [ ] --agentic is default behavior
- [ ] --deep flag triggers deep optimization mode
- [ ] Planner receives mode information

## Execution Strategy

### Wave 1: Read Current Flag Handling
**Task 1**: Read setup_model.sh argument parsing
- Find current flag handling code
- Understand how flags are passed to orchestrator
- Identify where to add --deep support

### Wave 2: Update Flag Logic
**Task 2**: Implement new flag behavior
- Change default values for optimize/agentic to "on"
- Add --deep flag parsing
- Update help text

### Wave 3: Update Orchestrator
**Task 3**: Modify orchestrator to accept and pass optimization mode
- Add `optimization_mode` parameter to `OpenCodeAgenticOrchestrator.__init__()`
- Add `optimization_mode` to `template_vars` in `_build_planner_prompt()`
- Pass mode when spawning orchestrator from setup_model.sh

### Wave 4: Test Changes
**Task 4**: Verify all components work together
- Test default run (automatic optimize + agentic)
- Test --deep flag triggers deep mode in planner
- Verify mode is correctly passed through all layers

## TODOs

- [ ] 1. Read setup_model.sh current flag handling

  **What to do**:
  - Read `/data/siraj/lm_start/setup_model.sh`
  - Find argument parsing section (likely getopts or manual parsing)
  - Identify optimize and agentic flag handling
  - Find where orchestrator is called

  **Acceptance Criteria**:
  - [ ] Current flag parsing understood
  - [ ] Orchestrator call location found
  - [ ] Default values identified

- [ ] 2. Update flag defaults and add --deep

  **What to do**:
  Modify `/data/siraj/lm_start/setup_model.sh`:
  
  1. Change defaults at lines 1008-1009:
     ```bash
     # OLD
     OPTIMIZE=false
     OPTIMIZE_AGENTIC=false
     
     # NEW
     OPTIMIZE=true
     OPTIMIZE_AGENTIC=true
     DEEP_MODE=false
     ```
  
  2. Add --deep flag to parsing (around line 1060 where other flags are parsed):
     ```bash
     --deep)
         DEEP_MODE=true
         shift
         ;;
     ```
  
  3. Update help text (around line 131-132):
     Add `--deep` to the help output
  
  4. Pass DEEP_MODE to orchestrator (around line 835 where OPTIMIZE_ARGS is built):
     ```bash
     if [ "$DEEP_MODE" = true ]; then
         OPTIMIZE_ARGS="$OPTIMIZE_ARGS --deep"
     fi
     ```

  **Acceptance Criteria**:
  - [ ] --optimize is now default (automatic)
  - [ ] --agentic is now default (automatic)
  - [ ] --deep flag added
  - [ ] Help text updated
  - [ ] Deep mode passed to orchestrator

- [ ] 3. Update orchestrator to accept and pass optimization mode

  **What to do**:
  Based on code analysis of `/data/siraj/lm_start/opencode_agentic_orchestrator.py`:
  
  1. In `__init__()` method (lines 51-76), add parameter:
     ```python
     def __init__(
         self,
         model_dir: str,
         model_id: str,
         goal: str = "maximize context",
         force: bool = False,
         min_experiments: int = 1,
         optimization_mode: str = "normal",  # NEW: "normal" or "deep"
     ):
         self.optimization_mode = optimization_mode  # Add this line
     ```
  
  2. In `_build_planner_prompt()` method (lines 221-300), add to template_vars dict:
     ```python
     template_vars = {
         # ... existing vars ...
         "optimization_mode": self.optimization_mode,
     }
     ```

  **Acceptance Criteria**:
  - [ ] Orchestrator accepts optimization_mode parameter
  - [ ] Mode passed to planner template via template_vars
  - [ ] Template variable {{optimization_mode}} is substituted correctly

- [ ] 4. Update setup_model.sh to pass mode to orchestrator

  **What to do**:
  In `/data/siraj/lm_start/setup_model.sh`, where orchestrator is instantiated:
  
  Find where opencode_agentic_orchestrator.py is called (around line 835 where OPTIMIZE_ARGS is built) and pass DEEP_MODE:
  ```bash
  # Add to OPTIMIZE_ARGS or create new arg
  if [ "$DEEP_MODE" = true ]; then
      OPTIMIZATION_MODE="deep"
  else
      OPTIMIZATION_MODE="normal"
  fi
  
  # Pass to orchestrator call
  python3 "$SCRIPT_DIR/opencode_agentic_orchestrator.py" \
      --model-dir "$MODEL_DIR" \
      --optimization-mode "$OPTIMIZATION_MODE" \
      # ... other args ...
  ```

  **Acceptance Criteria**:
  - [ ] DEEP_MODE value passed from setup_model.sh to orchestrator
  - [ ] Orchestrator receives correct mode

- [ ] 5. Test flag behavior

  **What to do**:
  Verify these work:
  - `./setup_model.sh model/name` (optimize + agentic automatic, normal mode)
  - `./setup_model.sh model/name --deep` (optimize + agentic, deep mode)
  - Check planner prompt file contains correct mode

  **Acceptance Criteria**:
  - [ ] Default run uses normal mode
  - [ ] --deep flag triggers deep mode
  - [ ] Planner receives mode in context

## Commit Strategy

- Commit: `feat(setup): enable optimize/agentic by default, add --deep flag`
- Files: `setup_model.sh`, `opencode_agentic_orchestrator.py`
- Message: "Make --optimize and --agentic default, add --deep mode support"

## Success Criteria

1. `./setup_model.sh model/name` runs with optimize + agentic by default
2. `--deep` flag enables deep optimization mode in planner
3. Planner receives correct optimization_mode in context
