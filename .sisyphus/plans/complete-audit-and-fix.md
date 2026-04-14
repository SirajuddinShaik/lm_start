# Complete Audit: lm-start vs setup_model.sh

## CRITICAL FINDINGS

### 1. Agentic Recovery Missing in `phase_generate_config`
**File:** `lm_start/phases/__init__.py`  
**Function:** `phase_generate_config()` (line 561)  
**Issue:** No agentic recovery while shell script has it  
**Impact:** When PM2 config generation fails, no recovery attempted

### 2. Phase Order Different
**setup_model.sh order:**
1. init
2. fetch_info
3. download
4. venv
5. smoke_test
6. extract_vllm_config
7. **generate_config**
8. **optimize**
9. finalize

**lm-start order:**
1. init
2. fetch_info
3. download
4. venv
5. smoke_test
6. extract_vllm_config
7. **generate_config**
8. **optimize** (optional)
9. finalize

**Status:** ✅ CORRECT - Matches when optimize enabled

### 3. Smoke Test Showing Wrong Status
**Issue:** Smoke test shows success but shouldn't have run without vLLM  
**Root Cause:** Smoke test checks for venv existence, not actual vLLM functionality

### 4. Extract Config Phase - Missing Context
**Issue:** `opencode_phase_agent.py` doesn't have `extract_vllm_config` in PHASE_CONTEXTS  
**Impact:** Agent doesn't know how to fix extraction errors

### 5. Error Messages Not Printed
**Issue:** Agentic recovery runs but user doesn't see progress  
**Fix:** Add print statements before calling agent

## FILES TO FIX

### 1. `lm_start/phases/__init__.py`
- [ ] Add agentic recovery to `phase_generate_config()` (around line 585)
- [ ] Add print statement before agent call in `phase_extract_vllm_config`
- [ ] Verify all error handlers have proper logging

### 2. `opencode_phase_agent.py`
- [ ] Add `extract_vllm_config` to PHASE_CONTEXTS dictionary
- [ ] Add `generate_config` to PHASE_CONTEXTS dictionary

### 3. Test Commands
```bash
# Test full flow with working model
lm-start setup TinyLlama/TinyLlama-1.1B-Chat-v0.4 -o -d

# Test with failing model to verify agent spawns
lm-start setup google/gemma-4-31B-it -o -d
```

## SPECIFIC FIXES NEEDED

### Fix 1: Add agentic recovery to phase_generate_config

**Location:** `lm_start/phases/__init__.py` line 561-607

Replace this:
```python
    result = run_script(pm2_script, [model_dir])
    if result.returncode != 0:
        if runner:
            runner.fail_phase(Phase.GENERATE_CONFIG, result.stderr)
        return PhaseResult(False, f"Failed to generate PM2 config: {result.stderr}")
```

With this:
```python
    result = run_script(pm2_script, [model_dir])
    if result.returncode != 0:
        error_msg = result.stderr or "Config generation failed"
        
        # Try agentic recovery
        print("[INFO] Attempting agentic recovery for generate_config...")
        if run_agentic_recovery(model_dir, "generate_config", error_msg):
            print("[INFO] OpenCode agent recovery succeeded, retrying config generation...")
            retry_result = run_script(pm2_script, [model_dir])
            if retry_result.returncode == 0:
                if runner:
                    runner.complete_phase(Phase.GENERATE_CONFIG)
                return PhaseResult(True, "Configs generated after agent fix")
        
        if runner:
            runner.fail_phase(Phase.GENERATE_CONFIG, result.stderr)
        return PhaseResult(False, f"Failed to generate PM2 config: {result.stderr}")
```

### Fix 2: Add missing phase contexts to opencode_phase_agent.py

**Location:** `opencode_phase_agent.py` lines 13-84

Add to PHASE_CONTEXTS:
```python
    "extract_vllm_config": """vLLM config extraction failed.
Common issues to check:
- Transformers version too old for model architecture
- Model files corrupted or incomplete
- Memory issues when loading model
- CUDA/PyTorch version mismatches
- Missing model weights or config files""",

    "generate_config": """PM2 and config generation failed.
Common issues to check:
- Missing template files
- Permission issues writing configs
- Invalid JSON/YAML in existing files
- Missing device configuration
- Template syntax errors""",
```

### Fix 3: Add debug output

Add print before every agent call to show it's attempting recovery.

## PRIORITY
CRITICAL - Blocks production use

## SUCCESS CRITERIA
- [ ] All 9 phases match setup_model.sh behavior
- [ ] Agentic recovery works in download, venv, smoke_test, extract_vllm_config, generate_config
- [ ] Error messages are clear and actionable
- [ ] User can see when agent is attempting recovery
