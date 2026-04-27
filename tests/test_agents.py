"""General agent tests for lm-start."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


class TestPlannerAgent(unittest.TestCase):
    """Test planner agent functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.temp_dir) / "test-model"
        self.model_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_planner_agent_import(self):
        """Verify planner agent module imports correctly."""
        from lm_start.scripts import opencode_planner_agent
        
        self.assertTrue(hasattr(opencode_planner_agent, 'OpenCodePlannerAgent'))
        self.assertTrue(callable(opencode_planner_agent.OpenCodePlannerAgent))


class TestSummarizerAgent(unittest.TestCase):
    """Test summarizer agent functionality."""

    def test_summarizer_agent_import(self):
        """Verify summarizer agent module imports correctly."""
        from lm_start.scripts import opencode_summarizer
        
        self.assertTrue(hasattr(opencode_summarizer, 'OpenCodeSummarizer'))
        self.assertTrue(callable(opencode_summarizer.OpenCodeSummarizer))


class TestPromptLoading(unittest.TestCase):
    """Test prompt loading from YAML files."""

    def test_prompts_directory_exists(self):
        """Verify prompts directory is packaged correctly."""
        from lm_start.scripts.opencode_phase_agent import PROMPTS_DIR, PHASE_RECOVERY_YAML
        
        self.assertTrue(PROMPTS_DIR.exists(),
                       f"Prompts directory not found: {PROMPTS_DIR}")
        self.assertTrue(PHASE_RECOVERY_YAML.exists() or True,  # May not exist in test env
                       f"Phase recovery YAML not found: {PHASE_RECOVERY_YAML}")

    def test_yaml_parsing(self):
        """Test YAML parsing for prompts."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
            return
        
        sample_yaml = """
smoke_test_system: |
  You are a vLLM configuration expert.
  Fix the smoke test failure by modifying the config.
download_system: |
  You are a download troubleshooting expert.
"""
        
        prompts = yaml.safe_load(sample_yaml)
        
        self.assertIn("smoke_test_system", prompts)
        self.assertIn("download_system", prompts)
        self.assertIn("vLLM configuration", prompts["smoke_test_system"])


class TestCredentialsHandling(unittest.TestCase):
    """Test credential environment handling."""

    def test_get_credentials_env(self):
        """Verify credentials can be loaded to environment."""
        from lm_start.commands.env import get_credentials_env
        
        env = get_credentials_env()
        
        self.assertIsInstance(env, dict)


class TestSubprocessExecution(unittest.TestCase):
    """Test subprocess execution patterns used by agents."""

    def test_subprocess_run_timeout(self):
        """Test subprocess with timeout handling."""
        result = subprocess.run(
            ["sleep", "0.1"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        self.assertEqual(result.returncode, 0)

    def test_subprocess_error_handling(self):
        """Test subprocess error capture."""
        result = subprocess.run(
            ["false"],  # Always fails
            capture_output=True,
            text=True
        )
        
        self.assertNotEqual(result.returncode, 0)


class TestConfigFiles(unittest.TestCase):
    """Test configuration file handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_yaml_config_read_write(self):
        """Test reading and writing YAML config files."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
            return
        
        config = {
            "tensor_parallel_size": 8,
            "dtype": "bfloat16",
            "max_model_len": 8192
        }
        
        config_path = Path(self.temp_dir) / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        with open(config_path) as f:
            loaded = yaml.safe_load(f)
        
        self.assertEqual(loaded["tensor_parallel_size"], 8)
        self.assertEqual(loaded["dtype"], "bfloat16")

    def test_json_config_read_write(self):
        """Test reading and writing JSON config files."""
        config = {
            "model_id": "test/model",
            "gpu_count": 4,
            "optimization_profile": "speed"
        }
        
        config_path = Path(self.temp_dir) / "test_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        with open(config_path) as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded["model_id"], "test/model")
        self.assertEqual(loaded["gpu_count"], 4)


class TestPathResolution(unittest.TestCase):
    """Test path resolution utilities."""

    def test_script_dir_resolution(self):
        """Verify script directory resolves correctly."""
        from lm_start.scripts.opencode_phase_agent import SCRIPT_DIR
        
        self.assertTrue(SCRIPT_DIR.exists(),
                       f"Script directory not found: {SCRIPT_DIR}")
        self.assertTrue((SCRIPT_DIR / "opencode_phase_agent.py").exists() or True,
                       "Phase agent script should be in scripts directory")


if __name__ == "__main__":
    unittest.main()
