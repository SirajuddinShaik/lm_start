#!/usr/bin/env python3
"""Manual test for phase agent to debug why it's not spawning opencode."""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lm_start.scripts.opencode_phase_agent import OpenCodePhaseAgent

# Test 1: Create agent and check if it can build prompt
print("="*60)
print("TEST 1: Building prompt")
print("="*60)

agent = OpenCodePhaseAgent(
    model_dir='/data/models/moonshotai-Kimi-K2.6',
    phase='smoke_test',
    error='vLLM process died during startup - CUDA OOM'
)
agent.attempts = 1

try:
    prompt = agent._build_prompt()
    print(f"✓ Prompt built: {len(prompt)} chars")
    
    # Check for unformatted placeholders
    if '{' in prompt and '}' in prompt:
        import re
        placeholders = re.findall(r'\{[^}]+\}', prompt)
        if placeholders:
            print(f"✗ Unformatted placeholders found: {placeholders[:5]}")
        else:
            print("✓ All placeholders formatted correctly")
    
    # Save prompt for inspection
    debug_file = Path('/tmp/phase_agent_prompt.txt')
    debug_file.write_text(prompt)
    print(f"✓ Prompt saved to: {debug_file}")
    
except Exception as e:
    print(f"✗ ERROR building prompt: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check if opencode binary exists
print("\n" + "="*60)
print("TEST 2: Checking OpenCode binary")
print("="*60)

from lm_start.scripts.opencode import OPENCODE_BINARY, is_installed

print(f"OpenCode binary path: {OPENCODE_BINARY}")
print(f"Binary exists: {OPENCODE_BINARY.exists()}")
print(f"Is executable: {OPENCODE_BINARY.exists() and OPENCODE_BINARY.stat().st_mode & 0o111}")
print(f"is_installed(): {is_installed()}")

if not is_installed():
    print("✗ OpenCode not installed! Run: lm-start init")
    sys.exit(1)

# Test 3: Try to spawn opencode manually
print("\n" + "="*60)
print("TEST 3: Spawning OpenCode manually")
print("="*60)

from lm_start.core.config import get_config
from lm_start.commands.env import get_credentials_env
from lm_start.scripts.opencode import OPENCODE_BINARY
import uuid

model_dir = Path('/data/models/moonshotai-Kimi-K2.6')
prompt_file = Path('/tmp/phase_agent_prompt.txt')
session_title = f"TestPhase-smoke_test-{uuid.uuid4().hex[:8]}"

cmd = [
    str(OPENCODE_BINARY),
    "run",
    "--model", "Grid/kimi-latest",
    "--agent", "build",
    "--dir", str(model_dir),
    "--title", session_title,
    "recovery",
    "--file", str(prompt_file),
]

print(f"Command: {' '.join(cmd)}")
print(f"Session title: {session_title}")

# Setup environment
env = get_config().get_env_dict(str(model_dir / ".venv"))
env.update(get_credentials_env())

print(f"\nEnvironment variables set:")
print(f"  VIRTUAL_ENV: {env.get('VIRTUAL_ENV', 'NOT SET')}")
print(f"  HF_TOKEN: {'SET' if env.get('HF_TOKEN') else 'NOT SET'}")
print(f"  PATH includes venv: {str(model_dir / '.venv' / 'bin') in env.get('PATH', '')}")

print("\nRunning opencode (timeout 60s)...")
try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    
    print(f"\nReturn code: {result.returncode}")
    print(f"\nSTDOUT (first 500 chars):")
    print(result.stdout[:500] if result.stdout else "(empty)")
    print(f"\nSTDERR (first 500 chars):")
    print(result.stderr[:500] if result.stderr else "(empty)")
    
    # Try to extract session ID
    import re
    match = re.search(r'ses_[a-zA-Z0-9]+', result.stdout + result.stderr)
    if match:
        print(f"\n✓ Session ID found: {match.group(0)}")
    else:
        print("\n✗ No session ID found in output")
        
except subprocess.TimeoutExpired:
    print("✗ TIMEOUT: OpenCode took too long to respond")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
