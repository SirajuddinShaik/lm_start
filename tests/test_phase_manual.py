"""Manual test for phase agent - run this to debug agent spawning issues.

This script creates a realistic test environment and runs the phase agent
to diagnose why fixes aren't being applied during smoke test failures.

Usage:
    /data/siraj/lm_start/.venv/bin/python tests/test_phase_manual.py

Or with specific model directory:
    python tests/test_phase_manual.py /path/to/model/dir
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_start.scripts.opencode_phase_agent import OpenCodePhaseAgent
from lm_start.scripts.opencode import OPENCODE_BINARY


def create_test_environment(base_dir: Path):
    """Create a realistic test model directory structure."""
    print(f"Creating test environment in: {base_dir}")
    
    # Create directories
    (base_dir / ".llm-context" / "model-context").mkdir(parents=True, exist_ok=True)
    (base_dir / ".opencode").mkdir(exist_ok=True)
    (base_dir / ".sessions").mkdir(exist_ok=True)
    
    # Model info (simulating Kimi-K2.6)
    model_info = {
        "model_id": "moonshotai/Kimi-K2.6",
        "num_parameters": "32000000000",
        "config": {
            "architectures": ["KimiForCausalLM"],
            "torch_dtype": "bfloat16"
        }
    }
    with open(base_dir / ".llm-context" / "model-context" / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Device config (simulating H200)
    device_config = {
        "gpus": {
            "count": 8,
            "memory_gb_per_gpu": 141,
            "names": ["NVIDIA H200"],
            "compute_capability": "9.0"
        },
        "cuda": {
            "version": "12.4",
            "available": True
        },
        "cpu": {
            "cores": 128,
            "memory_gb": 1024
        }
    }
    with open(base_dir / ".llm-context" / "model-context" / "device_config.json", "w") as f:
        json.dump(device_config, f, indent=2)
    
    # vLLM extracted config
    vllm_config = {
        "memory_estimate": {
            "parameters_b": 32.0,
            "total_size_gb": 64.0,
            "activation_memory_gb": 16.0
        },
        "is_moe": True,
        "hf_config": {
            "architectures": ["KimiForCausalLM"],
            "max_position_embeddings": 131072
        }
    }
    with open(base_dir / ".llm-context" / "model-context" / "vllm_extracted_config.json", "w") as f:
        json.dump(vllm_config, f, indent=2)
    
    # Create a deliberately broken smoke test config
    smoke_config = """tensor_parallel_size: 1
gpu_memory_utilization: 0.90
dtype: bfloat16
max_model_len: 65536
enforce_eager: false
enable_chunked_prefill: true
"""
    smoke_config_path = base_dir / "smoke_test_config.yaml"
    smoke_config_path.write_text(smoke_config)
    
    # Create a fake smoke test log showing OOM
    smoke_log = """INFO: Starting vLLM server...
INFO: Loading model weights...
WARNING: Model size (64GB) exceeds available VRAM with TP=1
ERROR: CUDA out of memory. Tried to allocate 40.00 GiB
ERROR: vLLM startup failed
"""
    (base_dir / "vllm_smoke_test.log").write_text(smoke_log)
    
    print("Test environment created successfully!")
    print(f"  Model: moonshotai/Kimi-K2.6")
    print(f"  GPUs: 8x H200 (141GB each)")
    print(f"  Config issue: tensor_parallel_size=1 (should be 8)")
    
    return base_dir


def test_phase_agent_direct():
    """Test phase agent directly with simulated smoke test failure."""
    print("\n" + "="*60)
    print("TEST: Phase Agent Direct Execution")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        create_test_environment(model_dir)
        
        # Create the agent
        error_msg = "CUDA out of memory. Tried to allocate 40.00 GiB"
        agent = OpenCodePhaseAgent(
            model_dir=str(model_dir),
            phase="smoke_test",
            error=error_msg,
            max_retries=1
        )
        
        print(f"\nAgent initialized:")
        print(f"  Model dir: {model_dir}")
        print(f"  Phase: smoke_test")
        print(f"  Error: {error_msg}")
        
        # Load and display model context
        print("\nLoading model context...")
        context = agent._load_model_context()
        print(f"  Model ID: {context.get('model_id')}")
        print(f"  GPU count: {context.get('gpu_count')}")
        print(f"  GPU memory: {context.get('gpu_memory')} GB")
        print(f"  Total VRAM: {context.get('total_gpu_memory_gb')} GB")
        print(f"  Model size: {context.get('model_size_gb')} GB")
        
        # Build and display prompt
        print("\nBuilding prompt...")
        prompt = agent._build_prompt()
        print(f"Prompt length: {len(prompt)} chars")
        print("\n--- Prompt Preview (first 500 chars) ---")
        print(prompt[:500])
        print("...")
        print("\n--- Prompt Preview (last 500 chars) ---")
        print(prompt[-500:])
        
        # Run the agent (this will actually spawn opencode)
        print("\n" + "="*60)
        print("EXECUTING AGENT (this may take up to 5 minutes)...")
        print("="*60)
        
        result = agent.run()
        
        print(f"\nAgent result: {result}")
        
        # Check if config was modified
        smoke_config = model_dir / "smoke_test_config.yaml"
        if smoke_config.exists():
            config_content = smoke_config.read_text()
            print("\nFinal smoke_test_config.yaml:")
            print(config_content)
            
            if "tensor_parallel_size: 8" in config_content:
                print("\n✓ SUCCESS: Config was updated to use 8 GPUs!")
            else:
                print("\n✗ FAILED: Config was NOT updated")
                print("  Current TP size is still 1, should be 8")
        
        return result


def test_opencode_cli_directly():
    """Test opencode CLI without the agent wrapper."""
    print("\n" + "="*60)
    print("TEST: OpenCode CLI Direct Test")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        create_test_environment(model_dir)
        
        # Create prompt file
        prompt = """Fix the smoke test failure by modifying smoke_test_config.yaml.

The error is: CUDA out of memory.

The current config has tensor_parallel_size: 1 but we have 8 GPUs available.
Change it to tensor_parallel_size: 8.

Execute this command:
sed -i 's/tensor_parallel_size: 1/tensor_parallel_size: 8/' smoke_test_config.yaml
"""
        prompt_file = model_dir / ".opencode" / "test_recovery.txt"
        prompt_file.write_text(prompt)
        
        print(f"Created prompt file: {prompt_file}")
        print(f"Working directory: {model_dir}")
        
        # Check config before
        config_before = (model_dir / "smoke_test_config.yaml").read_text()
        print("\nConfig BEFORE:")
        print(config_before)
        
        # Build opencode command exactly as agent does
        import uuid
        cmd = [
            str(OPENCODE_BINARY),
            "run",
            "--model", "Grid/kimi-latest",
            "--agent", "build",
            "--dir", str(model_dir),
            "--title", f"ManualTest-{uuid.uuid4().hex[:8]}",
            "recovery",
            "--file", str(prompt_file),
        ]
        
        print(f"\nExecuting command:")
        print(" \\n  ".join(cmd))
        
        # Run opencode
        print("\nRunning opencode (this may take 2-3 minutes)...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(model_dir)
        )
        
        print(f"\nReturn code: {result.returncode}")
        if result.stdout:
            print(f"\nSTDOUT:\n{result.stdout[:2000]}")
        if result.stderr:
            print(f"\nSTDERR:\n{result.stderr[:1000]}")
        
        # Check config after
        config_after = (model_dir / "smoke_test_config.yaml").read_text()
        print("\nConfig AFTER:")
        print(config_after)
        
        if config_before != config_after:
            print("\n✓ SUCCESS: Config was modified!")
        else:
            print("\n✗ FAILED: Config was not modified")


def test_sed_command_execution():
    """Test that sed command actually works for config modification."""
    print("\n" + "="*60)
    print("TEST: Sed Command Execution")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "smoke_test_config.yaml"
        
        # Write initial config
        initial = "tensor_parallel_size: 1\ngpu_memory_utilization: 0.90\n"
        config_path.write_text(initial)
        
        print(f"Initial config:\n{initial}")
        
        # Execute sed command
        cmd = f"sed -i 's/tensor_parallel_size: 1/tensor_parallel_size: 8/' {config_path}"
        print(f"\nExecuting: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        print(f"Return code: {result.returncode}")
        
        # Read result
        final = config_path.read_text()
        print(f"\nFinal config:\n{final}")
        
        if "tensor_parallel_size: 8" in final:
            print("\n✓ Sed command worked correctly!")
        else:
            print("\n✗ Sed command failed")


if __name__ == "__main__":
    print("OpenCode Phase Agent Manual Test Suite")
    print("=" * 60)
    
    # Check if opencode is installed
    if not OPENCODE_BINARY.exists():
        print(f"\nWARNING: OpenCode not found at {OPENCODE_BINARY}")
        print("Some tests will be skipped.")
    else:
        print(f"\nOpenCode found: {OPENCODE_BINARY}")
    
    # Run tests
    try:
        # Test 1: Simple sed command
        test_sed_command_execution()
        
        # Test 2: Direct opencode CLI
        if OPENCODE_BINARY.exists():
            test_opencode_cli_directly()
        
        # Test 3: Full agent workflow
        if OPENCODE_BINARY.exists():
            print("\n" + "="*60)
            print("Ready to run full agent test?")
            print("This will spawn an actual OpenCode agent and may take 5+ minutes.")
            response = input("Continue? (y/N): ")
            if response.lower() == 'y':
                test_phase_agent_direct()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
