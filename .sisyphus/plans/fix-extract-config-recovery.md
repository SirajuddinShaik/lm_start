# Fix: Add Agentic Recovery to phase_extract_vllm_config

## Problem
The `phase_extract_vllm_config()` function in `/data/siraj/lm_start2/lm_start/phases/__init__.py` does NOT have agentic recovery, while the shell script `setup_model.sh` does.

## Impact
When `extract_vllm_config.py` fails (e.g., due to transformers version issues), lm-start fails immediately without trying to recover, while setup_model.sh would spawn OpenCode agent to fix it.

## Fix Required

### File to Modify
`/data/siraj/lm_start2/lm_start/phases/__init__.py`

### Location
Around line 529-532 in `phase_extract_vllm_config()` function

### Current Code (lines 529-532)
```python
    if result.returncode != 0:
        if runner:
            runner.fail_phase(Phase.EXTRACT_VLLM_CONFIG, result.stderr)
        return PhaseResult(False, f"Failed to extract config: {result.stderr}")
```

### New Code (replace lines 529-532)
```python
    if result.returncode != 0:
        error_msg = result.stderr or "Extraction failed"
        
        # Try agentic recovery (same as setup_model.sh)
        if run_agentic_recovery(model_dir, "extract_vllm_config", error_msg):
            print("[INFO] OpenCode agent recovery succeeded, retrying extraction...")
            retry_result = subprocess.run(
                [str(python_exec), str(extract_script), model_dir, "--max-model-len", "1024"],
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
        return PhaseResult(False, f"Failed to extract config: {result.stderr}")
```

## Verification
After fix, run:
```bash
source /data/siraj/lm_start2/.venv/bin/activate
lm-start setup google/gemma-4-31B-it -o -d
```

When extract fails, should see:
```
[INFO] Attempting agentic recovery for extract_vllm_config...
[INFO] OpenCode agent recovery succeeded, retrying extraction...
```

## Priority
CRITICAL - This is the last phase missing agentic recovery
