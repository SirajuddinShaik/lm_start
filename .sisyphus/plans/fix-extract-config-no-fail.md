# URGENT FIX: Make phase_extract_vllm_config NOT fail

## Problem
`phase_extract_vllm_config()` in `lm_start/phases/__init__.py` returns `PhaseResult(False, ...)` when extraction fails, stopping the pipeline.

But `setup_model.sh` lines 765-769 shows it should WARN and CONTINUE:
```bash
else
    log_warn "vLLM config extraction failed (model may not be supported)"
    state_fail_phase "extract_vllm_config:Extraction failed"
    # Continue anyway - agent will fall back to guessing  <-- THIS IS KEY!
fi
```

## The Fix

**File:** `/data/siraj/lm_start2/lm_start/phases/__init__.py`

**Location:** Lines 529-566 (in `phase_extract_vllm_config` function)

**Current Code (WRONG - stops pipeline):**
```python
    if result.returncode != 0:
        error_msg = result.stderr or "Extraction failed"
        
        # Try agentic recovery (same as setup_model.sh)
        if run_agentic_recovery(model_dir, "extract_vllm_config", error_msg):
            print("[INFO] OpenCode agent recovery succeeded, retrying extraction...")
            retry_result = subprocess.run(
                [...],
                capture_output=True,
                text=True,
                env={**os.environ, **get_credentials_env()},
            )
            if retry_result.returncode == 0:
                if runner:
                    runner.complete_phase(Phase.EXTRACT_VLLM_CONFIG)
                return PhaseResult(True, "vLLM config extracted after agent fix")
        
        if runner:
            runner.fail_phase(Phase.EXTRACT_VLLM_CONFIG, result.stderr)
        return PhaseResult(False, f"Failed to extract config: {result.stderr}")  # <-- STOPS HERE!

    if runner:
        runner.complete_phase(Phase.EXTRACT_VLLM_CONFIG)
    return PhaseResult(True, "vLLM config extracted")
```

**New Code (CORRECT - matches setup_model.sh):**
```python
    if result.returncode != 0:
        print(f"[WARN] vLLM config extraction failed: {result.stderr[:200]}")
        print("[INFO] Continuing anyway - optimizer agent will fall back to guessing")
        if runner:
            runner.fail_phase(Phase.EXTRACT_VLLM_CONFIG, result.stderr)
        # DON'T RETURN FALSE - continue like setup_model.sh lines 765-769
        return PhaseResult(True, "Extraction failed but continuing - agent will handle")

    if runner:
        runner.complete_phase(Phase.EXTRACT_VLLM_CONFIG)
    return PhaseResult(True, "vLLM config extracted")
```

## Why This Matters
When extraction fails (e.g., gemma4 architecture not recognized), the optimize phase runs the agent which figures out the config through experimentation rather than extraction.

## Verification
After fix, run:
```bash
source /data/siraj/lm_start2/.venv/bin/activate
lm-start setup google/gemma-4-31B-it -o -d
```

Should see:
```
[WARN] vLLM config extraction failed: ...
[INFO] Continuing anyway - optimizer agent will fall back to guessing
✓ Extract vLLM Config
Running Optimize phase...
[INFO] Starting optimization experiments...
```

## Priority
URGENT - Blocking model deployment for new architectures
