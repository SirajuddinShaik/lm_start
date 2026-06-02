"""Tests for experiment_agent module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import json


class TestBenchmarkRegistry(unittest.TestCase):
    """Test benchmark registry functionality."""

    def test_registry_has_two_entries(self):
        """Verify benchmark registry has exactly 2 entries."""
        from lm_start.agents.experiment_agent import ExperimentAgent
        self.assertEqual(len(ExperimentAgent.BENCHMARK_REGISTRY), 2)

    def test_registry_has_comprehensive_and_stress(self):
        """Verify registry contains both comprehensive and stress_test."""
        from lm_start.agents.experiment_agent import ExperimentAgent

        names = [b['name'] for b in ExperimentAgent.BENCHMARK_REGISTRY]
        self.assertIn('comprehensive', names)
        self.assertIn('stress_test', names)

    def test_comprehensive_benchmark_config(self):
        """Test comprehensive benchmark has correct configuration."""
        from lm_start.agents.experiment_agent import ExperimentAgent

        comprehensive = next(
            b for b in ExperimentAgent.BENCHMARK_REGISTRY
            if b['name'] == 'comprehensive'
        )

        self.assertEqual(comprehensive['method'], 'run_comprehensive_benchmark')
        self.assertTrue(comprehensive['enabled'])
        self.assertIn('description', comprehensive)

    def test_stress_test_benchmark_config(self):
        """Test stress_test benchmark has correct configuration."""
        from lm_start.agents.experiment_agent import ExperimentAgent

        stress_test = next(
            b for b in ExperimentAgent.BENCHMARK_REGISTRY
            if b['name'] == 'stress_test'
        )

        self.assertEqual(stress_test['method'], 'run_stress_test')
        self.assertTrue(stress_test['enabled'])
        self.assertIn('description', stress_test)
        self.assertEqual(stress_test['target_percent'], 75)
        self.assertEqual(stress_test['seq_headroom'], 2048)


class TestRunBenchmarkIteration(unittest.TestCase):
    """Test benchmark iteration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.temp_dir) / "test-model"
        self.model_dir.mkdir(exist_ok=True)

        # Create required subdirectories
        (self.model_dir / ".llm-context" / "model-context").mkdir(parents=True, exist_ok=True)

        # Create minimal required files
        with open(self.model_dir / ".llm-context" / "model-context" / "model_info.json", "w") as f:
            json.dump({"model_id": "test/model", "vllm_compatible": True}, f)

        with open(self.model_dir / ".llm-context" / "model-context" / "device_config.json", "w") as f:
            json.dump({"gpus": {"count": 1, "memory_gb_per_gpu": 24, "name": "RTX4090"}}, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('lm_start.agents.experiment_agent.BenchmarkRunner')
    def test_runs_all_enabled_benchmarks(self, mock_runner_class):
        """Test that _run_benchmark iterates through registry and runs enabled benchmarks."""
        from lm_start.agents.experiment_agent import ExperimentAgent

        # Create mock runner
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        # Set up mock return values
        mock_runner.run_comprehensive_benchmark.return_value = {
            'summary': {'overall_success': True},
            'tests': {}
        }
        mock_runner.run_stress_test.return_value = {
            'crashed': False,
            'metrics': {'success_rate': 1.0}
        }

        # Create agent instance
        agent = ExperimentAgent(str(self.model_dir), enable_benchmarking=True)

        # Mock the BENCHMARK_REGISTRY to ensure we test iteration
        test_config = {'max_model_len': 32768}

        # Call _run_benchmark
        result = agent._run_benchmark(test_config, 8000)

        # Verify both benchmark methods were called
        mock_runner.run_comprehensive_benchmark.assert_called_once()
        mock_runner.run_stress_test.assert_called_once()

        # Verify stress_test was called with correct parameters
        call_kwargs = mock_runner.run_stress_test.call_args[1]
        self.assertEqual(call_kwargs['target_percent'], 0.75)
        self.assertEqual(call_kwargs['seq_headroom'], 2048)

    @patch('lm_start.agents.experiment_agent.BenchmarkRunner')
    def test_skips_disabled_benchmarks(self, mock_runner_class):
        """Test that disabled benchmarks are skipped."""
        from lm_start.agents.experiment_agent import ExperimentAgent

        # Create mock runner
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        # Create agent
        agent = ExperimentAgent(str(self.model_dir), enable_benchmarking=True)

        # Disable one benchmark
        original_registry = agent.BENCHMARK_REGISTRY.copy()
        agent.BENCHMARK_REGISTRY = [
            {
                'name': 'comprehensive',
                'method': 'run_comprehensive_benchmark',
                'enabled': False,  # Disabled
                'description': 'Disabled test'
            },
            {
                'name': 'stress_test',
                'method': 'run_stress_test',
                'enabled': True,
                'target_percent': 0.75,
                'seq_headroom': 2048
            }
        ]

        mock_runner.run_stress_test.return_value = {'crashed': False}

        try:
            agent._run_benchmark({'max_model_len': 32768}, 8000)

            # Comprehensive should not be called (disabled)
            mock_runner.run_comprehensive_benchmark.assert_not_called()
            # Stress test should be called
            mock_runner.run_stress_test.assert_called_once()
        finally:
            # Restore original
            agent.BENCHMARK_REGISTRY = original_registry

    @patch('lm_start.agents.experiment_agent.BenchmarkRunner')
    def test_returns_aggregated_results(self, mock_runner_class):
        """Test aggregated result structure from _run_benchmark."""
        from lm_start.agents.experiment_agent import ExperimentAgent

        # Create mock runner
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        # Set up mock return values
        mock_runner.run_comprehensive_benchmark.return_value = {
            'summary': {'overall_success': True},
            'tests': {'test1': {'success': True}}
        }
        mock_runner.run_stress_test.return_value = {
            'crashed': False,
            'metrics': {'success_rate': 0.95}
        }

        # Create agent
        agent = ExperimentAgent(str(self.model_dir), enable_benchmarking=True)
        test_config = {'max_model_len': 32768}

        result = agent._run_benchmark(test_config, 8000)

        # Verify aggregated result structure
        self.assertIn('config', result)
        self.assertIn('individual_results', result)
        self.assertIn('all_passed', result)
        self.assertIn('benchmarks_run', result)
        self.assertIn('benchmarks_passed', result)
        self.assertIn('timestamp', result)

        # Verify config is preserved
        self.assertEqual(result['config'], test_config)

        # Verify individual results are included
        self.assertIn('comprehensive', result['individual_results'])
        self.assertIn('stress_test', result['individual_results'])

        # Verify all_passed is True when both pass
        self.assertTrue(result['all_passed'])
        self.assertEqual(result['benchmarks_run'], ['comprehensive', 'stress_test'])
        self.assertEqual(result['benchmarks_passed'], ['comprehensive', 'stress_test'])

    @patch('lm_start.agents.experiment_agent.BenchmarkRunner')
    def test_aggregated_results_with_failures(self, mock_runner_class):
        """Test aggregated results when some benchmarks fail."""
        from lm_start.agents.experiment_agent import ExperimentAgent

        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        # One passes, one crashes
        mock_runner.run_comprehensive_benchmark.return_value = {
            'summary': {'overall_success': True}
        }
        mock_runner.run_stress_test.return_value = {
            'crashed': True,
            'metrics': {'success_rate': 0.5}
        }

        agent = ExperimentAgent(str(self.model_dir), enable_benchmarking=True)
        result = agent._run_benchmark({'max_model_len': 32768}, 8000)

        # all_passed should be False
        self.assertFalse(result['all_passed'])
        # Only comprehensive should be in passed list
        self.assertEqual(result['benchmarks_passed'], ['comprehensive'])

    @patch('lm_start.agents.experiment_agent.BenchmarkRunner')
    def test_handles_benchmark_exceptions(self, mock_runner_class):
        """Test exception handling in benchmark execution."""
        from lm_start.agents.experiment_agent import ExperimentAgent

        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        # Make one benchmark raise an exception
        mock_runner.run_comprehensive_benchmark.side_effect = Exception("Test error")
        mock_runner.run_stress_test.return_value = {'crashed': False}

        agent = ExperimentAgent(str(self.model_dir), enable_benchmarking=True)
        result = agent._run_benchmark({'max_model_len': 32768}, 8000)

        # Should handle exception gracefully
        self.assertIn('comprehensive', result['individual_results'])
        self.assertEqual(result['individual_results']['comprehensive']['error'], "Test error")
        self.assertTrue(result['individual_results']['comprehensive']['crashed'])

        # all_passed should be False
        self.assertFalse(result['all_passed'])

    @patch('lm_start.agents.experiment_agent.BenchmarkRunner')
    def test_quick_mode_passed_to_comprehensive(self, mock_runner_class):
        """Test quick_mode is passed to comprehensive benchmark."""
        from lm_start.agents.experiment_agent import ExperimentAgent

        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        mock_runner.run_comprehensive_benchmark.return_value = {'summary': {'overall_success': True}}
        mock_runner.run_stress_test.return_value = {'crashed': False}

        agent = ExperimentAgent(str(self.model_dir), enable_benchmarking=True, quick_mode=True)
        agent._run_benchmark({'max_model_len': 32768}, 8000)

        # Verify quick_mode was passed
        call_kwargs = mock_runner.run_comprehensive_benchmark.call_args[1]
        self.assertTrue(call_kwargs['quick_mode'])

    def test_run_benchmark_returns_none_when_disabled(self):
        """Test _run_benchmark returns None when benchmarking is disabled."""
        from lm_start.agents.experiment_agent import ExperimentAgent

        agent = ExperimentAgent(str(self.model_dir), enable_benchmarking=False)
        result = agent._run_benchmark({'max_model_len': 32768}, 8000)

        self.assertIsNone(result)


class TestBenchmarkRegistryMethodValidation(unittest.TestCase):
    """Test that benchmark registry methods exist on BenchmarkRunner."""

    def test_all_methods_exist_on_runner(self):
        """Verify all registered benchmark methods exist."""
        from lm_start.agents.experiment_agent import ExperimentAgent
        from lm_start.utils.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner('test-model')

        for bench_spec in ExperimentAgent.BENCHMARK_REGISTRY:
            method_name = bench_spec['method']
            self.assertTrue(
                hasattr(runner, method_name),
                f"Method '{method_name}' not found on BenchmarkRunner"
            )
            self.assertTrue(
                callable(getattr(runner, method_name)),
                f"'{method_name}' is not callable"
            )


if __name__ == '__main__':
    unittest.main()
