"""Tests for benchmark_runner module."""

import unittest
from unittest.mock import Mock, patch
from dataclasses import asdict
from lm_start.utils.benchmark_runner import BenchmarkRunner, BenchmarkResult, BenchmarkMetrics


class TestStressTest(unittest.TestCase):
    """Test stress test functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = BenchmarkRunner('test-model')

    def test_stress_test_calculates_correct_params(self):
        """Test parameter calculation for stress test."""
        # Test parameter calculation
        config = {'max_num_seqs': 192, 'max_model_len': 32768}

        # Should calculate: 144 concurrent (75% of 192), 30720 tokens (32768 - 2048)
        with patch.object(self.runner, 'run_serve_benchmark') as mock:
            mock.return_value = BenchmarkResult(
                config=config,
                metrics=BenchmarkMetrics(successful_requests=144, failed_requests=0, ttft_ms=3000),
                test_type='stress_test',
                timestamp='',
                raw_output='',
                success=True
            )

            result = self.runner.run_stress_test(config, 0.75, 2048)

            # Verify mock was called with correct params
            call_kwargs = mock.call_args[1]
            self.assertEqual(call_kwargs['num_prompts'], 144)
            self.assertEqual(call_kwargs['max_concurrency'], 144)
            self.assertEqual(call_kwargs['input_len'], 30720)
            self.assertEqual(call_kwargs['output_len'], 512)
            self.assertEqual(call_kwargs['test_name'], 'stress_test')

    def test_stress_test_defaults_to_75_percent(self):
        """Test stress test uses 75% default when not specified."""
        config = {'max_num_seqs': 100, 'max_model_len': 10000}

        with patch.object(self.runner, 'run_serve_benchmark') as mock:
            mock.return_value = BenchmarkResult(
                config=config,
                metrics=BenchmarkMetrics(successful_requests=75, failed_requests=0),
                test_type='stress_test',
                timestamp='',
                raw_output='',
                success=True
            )

            self.runner.run_stress_test(config)

            # Should use 0.75 (75%) as default
            call_kwargs = mock.call_args[1]
            self.assertEqual(call_kwargs['num_prompts'], 75)  # 75% of 100

    def test_stress_test_returns_crashed_field(self):
        """Test crashed field is correctly set in result."""
        config = {'max_num_seqs': 128, 'max_model_len': 32768}

        # Test successful case (should not crash)
        with patch.object(self.runner, 'run_serve_benchmark') as mock:
            mock.return_value = BenchmarkResult(
                config=config,
                metrics=BenchmarkMetrics(successful_requests=96, failed_requests=0),
                test_type='stress_test',
                timestamp='',
                raw_output='',
                success=True
            )

            result = self.runner.run_stress_test(config, 0.75, 2048)

            # Successful requests equal expected, no failures -> not crashed
            self.assertFalse(result['crashed'])

        # Test failed case (should mark as crashed)
        with patch.object(self.runner, 'run_serve_benchmark') as mock:
            mock.return_value = BenchmarkResult(
                config=config,
                metrics=BenchmarkMetrics(successful_requests=10, failed_requests=86),
                test_type='stress_test',
                timestamp='',
                raw_output='',
                success=False
            )

            result = self.runner.run_stress_test(config, 0.75, 2048)

            # More failed than successful -> crashed
            self.assertTrue(result['crashed'])

    def test_stress_test_includes_test_parameters(self):
        """Verify test_parameters are included in result."""
        config = {'max_num_seqs': 128, 'max_model_len': 32768}

        with patch.object(self.runner, 'run_serve_benchmark') as mock:
            mock.return_value = BenchmarkResult(
                config=config,
                metrics=BenchmarkMetrics(successful_requests=96, failed_requests=0),
                test_type='stress_test',
                timestamp='',
                raw_output='',
                success=True
            )

            result = self.runner.run_stress_test(config, 0.75, 2048)

            # Verify test_parameters structure
            self.assertIn('test_parameters', result)
            params = result['test_parameters']
            self.assertEqual(params['num_prompts'], 96)
            self.assertEqual(params['max_concurrency'], 96)
            self.assertEqual(params['input_len'], 30720)
            self.assertEqual(params['output_len'], 512)
            self.assertEqual(params['target_percent'], 0.75)
            self.assertEqual(params['seq_headroom'], 2048)

    def test_stress_test_metrics_calculation(self):
        """Test metrics are calculated correctly in result."""
        config = {'max_num_seqs': 100, 'max_model_len': 10000}

        with patch.object(self.runner, 'run_serve_benchmark') as mock:
            mock.return_value = BenchmarkResult(
                config=config,
                metrics=BenchmarkMetrics(
                    successful_requests=75,
                    failed_requests=5,
                    ttft_ms=2500.0,
                    tpot_ms=150.0,
                    throughput_tok_s=1000.0
                ),
                test_type='stress_test',
                timestamp='',
                raw_output='',
                success=True
            )

            result = self.runner.run_stress_test(config, 0.75, 500)

            # Verify metrics structure
            self.assertIn('metrics', result)
            metrics = result['metrics']
            self.assertAlmostEqual(metrics['success_rate'], 75/80)  # 75/(75+5)
            self.assertEqual(metrics['ttft_ms'], 2500.0)
            self.assertEqual(metrics['tpot_ms'], 150.0)
            self.assertEqual(metrics['throughput'], 1000.0)
            self.assertEqual(metrics['successful_requests'], 75)
            self.assertEqual(metrics['failed_requests'], 5)

    def test_stress_test_min_constraints(self):
        """Test stress test enforces minimum constraints."""
        # Very small max_num_seqs should still produce at least 1 concurrent
        config = {'max_num_seqs': 1, 'max_model_len': 1000}

        with patch.object(self.runner, 'run_serve_benchmark') as mock:
            mock.return_value = BenchmarkResult(
                config=config,
                metrics=BenchmarkMetrics(successful_requests=1, failed_requests=0),
                test_type='stress_test',
                timestamp='',
                raw_output='',
                success=True
            )

            result = self.runner.run_stress_test(config, 0.75, 2048)

            # Should ensure at least 1 prompt
            self.assertEqual(result['test_parameters']['num_prompts'], 1)
            # Should ensure at least 1024 input length
            self.assertGreaterEqual(result['test_parameters']['input_len'], 1024)


class TestBenchmarkResultStructure(unittest.TestCase):
    """Test BenchmarkResult dataclass and related structures."""

    def test_benchmark_result_creation(self):
        """Test BenchmarkResult can be created with required fields."""
        metrics = BenchmarkMetrics(
            throughput_tok_s=100.0,
            ttft_ms=500.0,
            successful_requests=64
        )

        result = BenchmarkResult(
            config={'test': 'config'},
            metrics=metrics,
            test_type='test',
            timestamp='2024-01-01T00:00:00',
            raw_output='test output',
            success=True
        )

        self.assertTrue(result.success)
        self.assertEqual(result.test_type, 'test')
        self.assertEqual(result.metrics.successful_requests, 64)

    def test_benchmark_result_asdict(self):
        """Test BenchmarkResult can be converted to dict."""
        metrics = BenchmarkMetrics(successful_requests=100, failed_requests=0)

        result = BenchmarkResult(
            config={'key': 'value'},
            metrics=metrics,
            test_type='test',
            timestamp='2024-01-01T00:00:00',
            raw_output='output',
            success=True
        )

        result_dict = asdict(result)

        self.assertEqual(result_dict['success'], True)
        self.assertEqual(result_dict['test_type'], 'test')
        self.assertEqual(result_dict['metrics']['successful_requests'], 100)


class TestRunServeBenchmark(unittest.TestCase):
    """Test run_serve_benchmark method."""

    def setUp(self):
        self.runner = BenchmarkRunner('test-model')

    def test_default_values_used_when_not_provided(self):
        """Test default values are used when parameters not specified."""
        with patch.object(self.runner, '_run_benchmark_command') as mock:
            mock.return_value = BenchmarkResult(
                config={},
                metrics=BenchmarkMetrics(),
                test_type='test',
                timestamp='',
                raw_output='',
                success=True
            )

            self.runner.run_serve_benchmark(
                config={},
                input_len=100,
                output_len=50,
                test_name='test'
            )

            # Verify defaults were used
            call_args = mock.call_args[0][0]  # First positional arg is cmd list
            cmd_str = ' '.join(call_args)

            # Check defaults in command
            self.assertIn('--num-prompts 64', cmd_str)  # DEFAULT_NUM_PROMPTS
            self.assertIn('--max-concurrency 32', cmd_str)  # DEFAULT_MAX_CONCURRENCY
            self.assertIn('--request-rate 32', cmd_str)  # DEFAULT_REQUEST_RATE

    def test_custom_values_override_defaults(self):
        """Test custom values override defaults."""
        with patch.object(self.runner, '_run_benchmark_command') as mock:
            mock.return_value = BenchmarkResult(
                config={},
                metrics=BenchmarkMetrics(),
                test_type='test',
                timestamp='',
                raw_output='',
                success=True
            )

            self.runner.run_serve_benchmark(
                config={},
                input_len=100,
                output_len=50,
                test_name='test',
                num_prompts=128,
                max_concurrency=64,
                request_rate=16
            )

            call_args = mock.call_args[0][0]
            cmd_str = ' '.join(call_args)

            self.assertIn('--num-prompts 128', cmd_str)
            self.assertIn('--max-concurrency 64', cmd_str)
            self.assertIn('--request-rate 16', cmd_str)


class TestParseServeOutput(unittest.TestCase):
    """Test parsing of vLLM bench serve output."""

    def setUp(self):
        self.runner = BenchmarkRunner('test-model')

    def test_parse_successful_requests(self):
        """Test parsing of successful requests count."""
        output = """
Some header text
Successful requests: 150
More text
"""
        metrics = self.runner._parse_serve_output(output)
        self.assertEqual(metrics.successful_requests, 150)

    def test_parse_failed_requests(self):
        """Test parsing of failed requests count."""
        output = """
Header
Failed requests: 10
Footer
"""
        metrics = self.runner._parse_serve_output(output)
        self.assertEqual(metrics.failed_requests, 10)

    def test_parse_throughput(self):
        """Test parsing of throughput metrics."""
        output = """
Output token throughput (tok/s): 1234.56
"""
        metrics = self.runner._parse_serve_output(output)
        self.assertAlmostEqual(metrics.throughput_tok_s, 1234.56)

    def test_parse_ttft(self):
        """Test parsing of TTFT metrics."""
        output = """
Mean TTFT (ms): 2500.0
Median TTFT (ms): 2400.0
P99 TTFT (ms): 3500.0
"""
        metrics = self.runner._parse_serve_output(output)
        self.assertAlmostEqual(metrics.ttft_ms, 2500.0)
        self.assertAlmostEqual(metrics.latency_p50_ms, 2400.0)
        self.assertAlmostEqual(metrics.latency_p99_ms, 3500.0)

    def test_parse_total_requests_calculated(self):
        """Test total requests is calculated from successful + failed."""
        output = """
Successful requests: 140
Failed requests: 10
"""
        metrics = self.runner._parse_serve_output(output)
        self.assertEqual(metrics.total_requests, 150)

    def test_parse_empty_output(self):
        """Test parsing handles empty output gracefully."""
        metrics = self.runner._parse_serve_output('')
        self.assertEqual(metrics.successful_requests, 0)
        self.assertEqual(metrics.failed_requests, 0)
        self.assertEqual(metrics.total_requests, 0)


if __name__ == '__main__':
    unittest.main()
