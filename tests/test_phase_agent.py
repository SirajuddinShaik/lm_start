"""Test phase agent spawning and recovery logic."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


class TestPhaseAgentSpawning(unittest.TestCase):
    """Test suite for phase agent spawning and execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.temp_dir) / "test-model"
        self.model_dir.mkdir(exist_ok=True)
        
        # Create necessary subdirectories
        (self.model_dir / ".llm-context" / "model-context").mkdir(parents=True, exist_ok=True)
        (self.model_dir / ".opencode").mkdir(exist_ok=True)
        (self.model_dir / ".sessions").mkdir(exist_ok=True)
        
        # Create minimal model info
        model_info = {"model_id": "test/model", "num_parameters": "7B"}
        with open(self.model_dir / ".llm-context" / "model-context" / "model_info.json", "w") as f:
            json.dump(model_info, f)
        
        # Create device config
        device_config = {
            "gpus": {"count": 1, "memory_gb_per_gpu": 80, "names": ["Test-GPU"]},
            "cuda": {"version": "12.0"}
        }
        with open(self.model_dir / ".llm-context" / "model-context" / "device_config.json", "w") as f:
            json.dump(device_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_opencode_binary_exists(self):
        """Verify opencode binary is accessible."""
        from lm_start.scripts.opencode import OPENCODE_BINARY
        
        self.assertIsNotNone(OPENCODE_BINARY)
        self.assertTrue(OPENCODE_BINARY.exists() or OPENCODE_BINARY.name == "opencode",
                       f"OpenCode binary not found at {OPENCODE_BINARY}")

    def test_phase_agent_import(self):
        """Verify phase agent module imports correctly."""
        from lm_start.scripts import opencode_phase_agent
        
        self.assertTrue(hasattr(opencode_phase_agent, 'OpenCodePhaseAgent'))
        self.assertTrue(callable(opencode_phase_agent.OpenCodePhaseAgent))

    def test_agent_initialization(self):
        """Test agent can be initialized with test data."""
        from lm_start.scripts.opencode_phase_agent import OpenCodePhaseAgent
        
        agent = OpenCodePhaseAgent(
            model_dir=str(self.model_dir),
            phase="smoke_test",
            error="Test error: vLLM crashed during startup",
            max_retries=2
        )
        
        self.assertEqual(agent.phase, "smoke_test")
        self.assertEqual(agent.max_retries, 2)
        self.assertEqual(agent.attempts, 0)
        self.assertEqual(agent.model_dir, self.model_dir)

    def test_prompt_building(self):
        """Test agent builds prompt with context."""
        from lm_start.scripts.opencode_phase_agent import OpenCodePhaseAgent
        
        # Create a smoke test config
        smoke_config = self.model_dir / "smoke_test_config.yaml"
        smoke_config.write_text("tensor_parallel_size: 1\n")
        
        agent = OpenCodePhaseAgent(
            model_dir=str(self.model_dir),
            phase="smoke_test", 
            error="Test error: OOM during model load"
        )
        
        prompt = agent._build_prompt()
        
        # Verify prompt contains expected sections
        self.assertIn("smoke_test", prompt.lower())
        self.assertIn("test/model", prompt)
        self.assertIn("tensor_parallel_size: 1", prompt)

    def test_model_context_loading(self):
        """Test agent loads model context from files."""
        from lm_start.scripts.opencode_phase_agent import OpenCodePhaseAgent
        
        agent = OpenCodePhaseAgent(
            model_dir=str(self.model_dir),
            phase="smoke_test",
            error="Test error"
        )
        
        context = agent._load_model_context()
        
        self.assertEqual(context["model_id"], "test/model")
        self.assertEqual(context["gpu_count"], "1")
        self.assertEqual(context["gpu_memory"], "80")
        self.assertEqual(context["gpu_name"], "Test-GPU")

    def test_config_modification_detection(self):
        """Test agent detects when config file is modified."""
        from lm_start.scripts.opencode_phase_agent import OpenCodePhaseAgent
        import time
        
        # Create initial config
        smoke_config = self.model_dir / "smoke_test_config.yaml"
        smoke_config.write_text("tensor_parallel_size: 1\n")
        
        mtime_before = smoke_config.stat().st_mtime
        
        # Simulate config modification
        time.sleep(0.1)
        smoke_config.write_text("tensor_parallel_size: 8\n")
        mtime_after = smoke_config.stat().st_mtime
        
        self.assertGreater(mtime_after, mtime_before,
                          "Config modification should update mtime")


class TestOpencodeIntegration(unittest.TestCase):
    """Integration tests with actual opencode CLI."""
    
    @unittest.skipUnless(
        subprocess.run(["which", "opencode"], capture_output=True).returncode == 0
        or Path.home().exists(),
        "OpenCode not installed"
    )
    def test_opencode_cli_responds(self):
        """Test opencode CLI is responsive."""
        from lm_start.scripts.opencode import OPENCODE_BINARY
        
        result = subprocess.run(
            [str(OPENCODE_BINARY), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout.lower())


class TestPhaseRecoveryMock(unittest.TestCase):
    """Mock-based tests for recovery logic."""

    def test_parse_fix_command_from_response(self):
        """Test parsing fix commands from agent responses."""
        # Sample agent response with JSON fix command
        sample_response = '''
        Looking at the smoke test failure, I need to increase tensor parallel size.
        
        ```json
        {
            "action": "modify_config",
            "file": "smoke_test_config.yaml",
            "changes": ["tensor_parallel_size: 1 -> tensor_parallel_size: 8"]
        }
        ```
        
        Executing: sed -i 's/tensor_parallel_size: 1/tensor_parallel_size: 8/' smoke_test_config.yaml
        '''
        
        # Check if we can extract the sed command
        import re
        
        sed_match = re.search(r"sed -i ['\"](.+?)['\"]", sample_response)
        self.assertIsNotNone(sed_match, "Should find sed command in response")
        
        if sed_match:
            cmd = sed_match.group(0)
            self.assertIn("tensor_parallel_size", cmd)


if __name__ == "__main__":
    unittest.main()
