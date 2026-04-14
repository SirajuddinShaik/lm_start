# Redesign Planner Prompt: Normal vs Deep Mode

## TL;DR

Redesign `/data/siraj/lm_start/prompts/opencode_planner.yaml` to support two optimization modes:
- **NORMAL mode**: Quick optimization (3-5 runs), get working system fast
- **DEEP mode** (--deep flag): Thorough research on backends, CUDA graphs, parameter sweeps

Remove early FINISH trigger when max_model_len == max_position_embeddings. Ensure agent optimizes progressively from current working level instead of random changes.

## Context

Current prompt has issues:
1. Early FINISH when max_model_len == max_position_embeddings (stops too early)
2. No mode differentiation - same prompt for quick vs thorough optimization
3. Agent randomly reduces max_model_len instead of building from working configs
4. Missing systematic backend/CUDA graph exploration

## Work Objectives

### Core Objective
Redesign planner prompt with split normal/deep mode structure and progressive optimization strategy.

### Concrete Deliverables
- Updated `opencode_planner.yaml` with:
  - `mode_detection` section for normal vs deep
  - `normal_mode_framework` for quick optimization
  - `deep_mode_framework` for thorough research
  - `advanced_optimizations` section (backend testing, CUDA graphs)
- Removed early FINISH trigger
- Clear "optimize from current level" guidance

### Definition of Done
- [ ] Prompt has clear normal vs deep mode sections
- [ ] Agent no longer finishes early at max_position_embeddings
- [ ] Agent builds progressively from working configs
- [ ] Deep mode includes backend and CUDA graph research

## Execution Strategy

### Wave 1: Read Current Prompt
**Task 1**: Read current `/data/siraj/lm_start/prompts/opencode_planner.yaml`
- Understand current structure
- Identify sections to modify
- Note template variables used

### Wave 2: Create New Prompt Structure
**Task 2**: Write redesigned prompt to file
- Add `mode_detection` section at top using `{{optimization_mode}}` template variable
- Create separate `normal_mode_framework` and `deep_mode_framework` sections
- Remove early FINISH logic from decision frameworks
- Add `advanced_optimizations` section for deep mode
- Update `critical_rules` to emphasize progressive optimization
- Update `output_format` with mode indication in reasoning

Note: Orchestrator passes variables via `template_vars` dict in `_build_planner_prompt()` method. Add `optimization_mode` to this dict based on --deep flag.

### Wave 3: Verify Integration
**Task 3**: Check prompt loads correctly
- Verify YAML syntax
- Check template variables match orchestrator
- Ensure sections are properly referenced

## TODOs

- [ ] 1. Read current planner prompt structure

  **What to do**:
  - Read `/data/siraj/lm_start/prompts/opencode_planner.yaml`
  - Document current sections and structure
  - Identify all template variables used

  **Acceptance Criteria**:
  - [ ] Current prompt structure understood
  - [ ] Template variables catalogued
  - [ ] Sections to modify identified

- [ ] 2. Create redesigned prompt with normal/deep modes

  **What to do**:
  Write new `opencode_planner.yaml` with these sections:
  
  1. **mode_detection**: Template section that shows current mode
  
  2. **planner_system**: Updated to reference mode
  
  3. **normal_mode_framework**:
     - Quick optimization (3-5 runs goal)
     - "Optimize from current" instead of random changes
     - NO early FINISH at max_position_embeddings
     - Incremental improvements only
  
  4. **deep_mode_framework**:
     - Thorough backend research priority
     - CUDA graph optimization priority
     - Systematic parameter sweeps
     - Performance ceiling testing
  
  5. **error_patterns**: Updated with FP8 warning
  
  6. **advanced_optimizations** (deep mode only):
     - Attention backend comparison (FLASH_ATTN, FLASHINFER, XFORMERS)
     - Distributed executor backend (mp, ray)
     - CUDA graph tuning (max_cudagraph_capture_size, compilation_config)
     - KV cache tuning options
     - Batching strategy sweeps
  
  7. **task_instructions**: Split by mode
  
  8. **critical_rules**:
     - Rule 1: "OPTIMIZE FROM CURRENT LEVEL - NEVER RANDOMLY REDUCE!"
     - Rule 4: "NO EARLY FINISH!"
  
  9. **output_format**: Updated JSON format with mode in reasoning

  **Acceptance Criteria**:
  - [ ] Normal mode framework emphasizes quick wins from current level
  - [ ] Deep mode framework includes backend/CUDA graph research
  - [ ] Early FINISH logic removed
  - [ ] Progressive optimization guidance added
  - [ ] Template variables preserved

- [ ] 3. Verify prompt integration

  **What to do**:
  - Check YAML syntax validity
  - Verify all template variables match orchestrator expectations
  - Ensure sections are properly formatted

  **Acceptance Criteria**:
  - [ ] YAML parses without errors
  - [ ] All template variables accounted for
  - [ ] Sections properly structured

## Commit Strategy

- Commit: `feat(planner): redesign prompt with normal/deep modes`
- Files: `prompts/opencode_planner.yaml`
- Message: "Add split optimization modes, remove early finish, add progressive optimization"

## Success Criteria

1. Prompt supports normal (quick) and deep (thorough) modes
2. Agent builds progressively from working configs
3. No early FINISH at max_position_embeddings
4. Deep mode includes systematic backend/CUDA graph research
5. All existing template variables preserved
