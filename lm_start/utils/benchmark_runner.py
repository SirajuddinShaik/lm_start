"""
Benchmark Runner for vLLM Configuration Testing

Uses only vllm bench serve for all benchmarks.
"""

import json
import os
import subprocess
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics


@dataclass
class BenchmarkMetrics:
    """Benchmark metrics for a single test."""

    throughput_tok_s: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    batch_size_avg: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    duration_seconds: float = 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    config: Dict[str, Any]
    metrics: BenchmarkMetrics
    test_type: str
    timestamp: str
    raw_output: str
    success: bool
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Comparison between two benchmark results."""

    baseline_context: int
    candidate_context: int
    throughput_diff_pct: float
    latency_diff_pct: float
    winner: str  # 'baseline', 'candidate', 'tie'
    improvement_pct: float
    recommendation: str


class BenchmarkRunner:
    """
    Run vLLM benchmarks using only vllm bench serve.
    """

    DEFAULT_NUM_PROMPTS = 64
    DEFAULT_MAX_CONCURRENCY = 32
    DEFAULT_REQUEST_RATE = 32
    DEFAULT_TIMEOUT = 1800  # 30 minutes

    def __init__(
        self,
        model_id: str,
        vllm_port: Optional[int] = None,
        venv_python: Optional[Path] = None,
        hf_home: Optional[str] = None,
        cuda_home: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model_id = model_id
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

        # Load from config if available, otherwise use defaults
        try:
            from core.config import get_config

            config = get_config()
            self.vllm_port = vllm_port or config.vllm_defaults.port
            self.hf_home = hf_home or config.paths.hf_home
            self.cuda_home = cuda_home or config.device.cuda.home
            # Find venv python relative to script or use default
            script_dir = Path(__file__).parent.parent
            self.venv_python = venv_python or (script_dir / ".venv" / "bin" / "python")
        except Exception:
            # Fallback to defaults if config not available
            self.vllm_port = vllm_port or 8000
            self.hf_home = hf_home or os.environ.get(
                "HF_HOME", "/data/.cache/huggingface"
            )
            self.cuda_home = cuda_home or "/usr/local/cuda-12.9"
            self.venv_python = venv_python or Path(
                "/data/siraj/lm_start/.venv/bin/python"
            )

    def log(self, message: str, level: str = "info"):
        """Log a message."""
        from rich.console import Console as _C
        _c = _C()
        if level == "error":
            _c.print(f"  [red]✗[/red]  {message}")
        elif level == "warning":
            _c.print(f"  [yellow]⚠[/yellow]  {message}")
        elif level == "success":
            _c.print(f"  [green]✓[/green]  {message}")
        else:
            _c.print(f"  [dim]{message}[/dim]")

    def run_serve_benchmark(
        self,
        config: Dict[str, Any],
        input_len: int,
        output_len: int,
        test_name: str,
        num_prompts: int = None,
        max_concurrency: int = None,
        request_rate: int = None,
        timeout: int = None,
    ) -> BenchmarkResult:
        """
        Run vLLM bench serve with static parameters.

        Args:
            config: vLLM configuration
            input_len: Input sequence length
            output_len: Output sequence length
            test_name: Name of the test
            num_prompts: Number of prompts (default: 100)
            max_concurrency: Max concurrent requests (default: 32)
            request_rate: Request rate per second (default: 32)
            timeout: Timeout in seconds (default: 1800 = 30 min)

        Returns:
            BenchmarkResult with metrics
        """
        # Use static defaults
        num_prompts = num_prompts or self.DEFAULT_NUM_PROMPTS
        max_concurrency = max_concurrency or self.DEFAULT_MAX_CONCURRENCY
        request_rate = request_rate or self.DEFAULT_REQUEST_RATE
        timeout = timeout or self.DEFAULT_TIMEOUT

        self.log(
            f"Running {test_name}: input={input_len}, output={output_len}, "
            f"prompts={num_prompts}, concurrency={max_concurrency}, rate={request_rate}, timeout={timeout}s"
        )

        cmd = [
            str(self.venv_python).replace("/python", "/vllm"),
            "bench",
            "serve",
            "--model",
            self.model_id,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(input_len),
            "--random-output-len",
            str(output_len),
            "--num-prompts",
            str(num_prompts),
            "--max-concurrency",
            str(max_concurrency),
            "--request-rate",
            str(request_rate),
            "--port",
            str(self.vllm_port),
            "--trust-remote-code",
        ]

        return self._run_benchmark_command(
            cmd,
            config,
            test_name,
            timeout,
            input_len=input_len,
            output_len=output_len,
        )

    def _run_benchmark_command(
        self,
        cmd: List[str],
        config: Dict[str, Any],
        test_name: str,
        timeout: int,
        **kwargs,
    ) -> BenchmarkResult:
        """Execute benchmark command and parse results."""
        env = os.environ.copy()
        env["HF_HOME"] = self.hf_home
        env["CUDA_HOME"] = self.cuda_home

        start_time = time.time()
        stdout_output = ""
        stderr_output = ""

        try:
            # Run the benchmark from model directory
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                cwd=str(self.venv_python.parent.parent.parent),
            )

            try:
                stdout_output, stderr_output = process.communicate(timeout=timeout)
                duration = time.time() - start_time

                if process.returncode != 0:
                    return BenchmarkResult(
                        config=config,
                        metrics=BenchmarkMetrics(),
                        test_type=test_name,
                        timestamp=datetime.now().isoformat(),
                        raw_output=stdout_output + "\n" + stderr_output,
                        success=False,
                        error=f"Process exited with code {process.returncode}: {stderr_output[:500]}",
                    )

                # Parse metrics from output
                metrics = self._parse_serve_output(stdout_output + stderr_output)
                metrics.duration_seconds = duration

                # Determine success based on failed requests
                success = (
                    metrics.failed_requests == 0 and metrics.successful_requests > 0
                )

                return BenchmarkResult(
                    config=config,
                    metrics=metrics,
                    test_type=test_name,
                    timestamp=datetime.now().isoformat(),
                    raw_output=stdout_output + "\n" + stderr_output,
                    success=success,
                    error=None
                    if success
                    else f"Failed requests: {metrics.failed_requests}",
                )

            except subprocess.TimeoutExpired:
                process.kill()
                return BenchmarkResult(
                    config=config,
                    metrics=BenchmarkMetrics(),
                    test_type=test_name,
                    timestamp=datetime.now().isoformat(),
                    raw_output="Timeout",
                    success=False,
                    error=f"Benchmark timed out after {timeout}s",
                )

        except Exception as e:
            return BenchmarkResult(
                config=config,
                metrics=BenchmarkMetrics(),
                test_type=test_name,
                timestamp=datetime.now().isoformat(),
                raw_output=f"Exception: {str(e)}\n{stderr_output}",
                success=False,
                error=str(e),
            )

    def _parse_serve_output(self, output: str) -> BenchmarkMetrics:
        """Parse vllm bench serve output to extract metrics."""
        metrics = BenchmarkMetrics()

        lines = output.split("\n")

        for line in lines:
            line = line.strip()

            # Parse successful requests
            if "Successful requests:" in line:
                try:
                    metrics.successful_requests = int(line.split(":")[1].strip())
                except:
                    pass

            # Parse failed requests
            elif "Failed requests:" in line:
                try:
                    metrics.failed_requests = int(line.split(":")[1].strip())
                except:
                    pass

            elif "Maximum request concurrency:" in line:
                try:
                    metrics.batch_size_avg = float(line.split(":")[1].strip())
                except:
                    pass

            # Parse output token throughput
            elif "Output token throughput (tok/s):" in line:
                try:
                    metrics.throughput_tok_s = float(line.split(":")[1].strip())
                except:
                    pass

            # Parse Mean TTFT
            elif "Mean TTFT (ms):" in line:
                try:
                    metrics.ttft_ms = float(line.split(":")[1].strip())
                except:
                    pass

            elif "Median TTFT (ms):" in line:
                try:
                    metrics.latency_p50_ms = float(line.split(":")[1].strip())
                except:
                    pass

            elif "P99 TTFT (ms):" in line:
                try:
                    metrics.latency_p99_ms = float(line.split(":")[1].strip())
                except:
                    pass

            # Parse Mean TPOT
            elif "Mean TPOT (ms):" in line:
                try:
                    metrics.tpot_ms = float(line.split(":")[1].strip())
                except:
                    pass

        # Calculate total requests
        metrics.total_requests = metrics.successful_requests + metrics.failed_requests

        return metrics

    def run_comprehensive_benchmark(
        self, config: Dict[str, Any], quick_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Run benchmark tests on a configuration using only bench serve.
        Tests are selected based on the configured max_model_len:
        - For max_model_len < 65536: Run tests at 32K and 64K input lengths
        - For max_model_len >= 65536: Run tests at 32K and 64K only (don't scale to full context)
        This ensures benchmarks complete within timeout even for very long context models.
        """
        self.log(
            f"Starting comprehensive benchmark for context={config.get('max_model_len')}"
        )

        results = {
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {},
        }

        max_context = config.get("max_model_len", 32768)

        # Select tests based on max_model_len
        # Only run a test if the model's max_context can handle it
        selected_tests = []

        # Always try 32K if model supports it
        if max_context >= 32768:
            selected_tests.append(
                {
                    "name": "ctx_32k",
                    "input_len": 32768,
                    "output_len": 500,
                }
            )

        # Only run 64K if model supports it and we're not in quick_mode
        if max_context >= 65536 and not quick_mode:
            selected_tests.append(
                {
                    "name": "ctx_64k",
                    "input_len": 65536,
                    "output_len": 500,
                }
            )

        # If model has very short context, use a smaller test
        if max_context < 32768:
            # Scale down for short context models
            input_len = min(8192, max_context // 2)
            output_len = min(500, max_context // 8)
            selected_tests.append(
                {
                    "name": f"ctx_{input_len}",
                    "input_len": input_len,
                    "output_len": output_len,
                }
            )

        if not selected_tests:
            self.log("No tests selected - max_model_len too small", "error")
            results["summary"] = self._calculate_summary(results["tests"])
            return results

        self.log(
            f"Selected {len(selected_tests)} test(s) for max_context={max_context}"
        )

        for test in selected_tests:
            result = self.run_serve_benchmark(
                config,
                input_len=test["input_len"],
                output_len=test["output_len"],
                test_name=test["name"],
                # Use static defaults
                num_prompts=self.DEFAULT_NUM_PROMPTS,
                max_concurrency=self.DEFAULT_MAX_CONCURRENCY,
                request_rate=self.DEFAULT_REQUEST_RATE,
                timeout=self.DEFAULT_TIMEOUT,
            )
            results["tests"][test["name"]] = asdict(result)

        # Calculate summary
        results["summary"] = self._calculate_summary(results["tests"])

        return results

    def run_stress_test(
        self,
        config: Dict[str, Any],
        target_percent: float = 0.75,
        seq_headroom: int = 2048
    ) -> Dict[str, Any]:
        """
        Run stress test at high concurrency x high sequence length.

        Args:
            config: vLLM configuration with max_num_seqs and max_model_len
            target_percent: Target concurrency as % of max_num_seqs (default 0.75)
            seq_headroom: Tokens to reserve from max_model_len (default 2048)

        Returns:
            Dict with test result and verdict
        """
        max_num_seqs = config.get('max_num_seqs', 128)
        max_model_len = config.get('max_model_len', 32768)

        # Calculate stress parameters
        target_concurrency = max(1, int(max_num_seqs * target_percent))
        target_seq_len = max(1024, max_model_len - seq_headroom)

        self.log(f"Running stress test: {target_concurrency} concurrent @ {target_seq_len} tokens")

        # Run single benchmark
        result = self.run_serve_benchmark(
            config=config,
            input_len=target_seq_len,
            output_len=500,  # Short output for quick test
            test_name="stress_test",
            num_prompts=target_concurrency,  # Saturate concurrency
            max_concurrency=target_concurrency,
                request_rate=target_concurrency,
                timeout=900,
            )

        # Collect metrics for agent analysis
        metrics = result.metrics
        success_rate = metrics.successful_requests / max(metrics.total_requests, 1)

        # Crashed = obvious fail. Otherwise agent analyzes all run results to decide.
        crashed = not result.success or metrics.failed_requests > metrics.successful_requests

        return {
            "test_type": "stress_test",
            "config": config,
            "raw_result": asdict(result),
            "crashed": crashed,
            "test_parameters": {
                "num_prompts": target_concurrency,
                "max_concurrency": target_concurrency,
                "input_len": target_seq_len,
                "output_len": 500,
                "target_percent": target_percent,
                "seq_headroom": seq_headroom,
            },
            "metrics": {
                "success_rate": success_rate,
                "ttft_ms": metrics.ttft_ms,
                "tpot_ms": metrics.tpot_ms,
                "throughput": metrics.throughput_tok_s,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_summary(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from test results."""
        summary = {
            "overall_success": True,
            "total_tests": len(tests),
            "successful_tests": 0,
            "failed_tests": 0,
            "avg_throughput": 0.0,
            "avg_ttft_ms": 0.0,
            "avg_tpot_ms": 0.0,
            "best_test": None,
            "worst_test": None,
        }

        throughputs = []
        ttfts = []
        tpots = []
        test_scores = {}

        for test_name, test_data in tests.items():
            if isinstance(test_data, list):
                # Multiple results (e.g., latency at different input lengths)
                for item in test_data:
                    if item.get("success"):
                        summary["successful_tests"] += 1
                        metrics = item.get("metrics", {})
                        if metrics.get("throughput_tok_s"):
                            throughputs.append(metrics["throughput_tok_s"])
                        if metrics.get("ttft_ms"):
                            ttfts.append(metrics["ttft_ms"])
                        if metrics.get("tpot_ms"):
                            tpots.append(metrics["tpot_ms"])
                    else:
                        summary["failed_tests"] += 1
                        summary["overall_success"] = False
            else:
                # Single result
                if test_data.get("success"):
                    summary["successful_tests"] += 1
                    metrics = test_data.get("metrics", {})
                    throughput = metrics.get("throughput_tok_s", 0)
                    if throughput:
                        throughputs.append(throughput)
                        test_scores[test_name] = throughput
                    if metrics.get("ttft_ms"):
                        ttfts.append(metrics["ttft_ms"])
                    if metrics.get("tpot_ms"):
                        tpots.append(metrics["tpot_ms"])
                else:
                    summary["failed_tests"] += 1
                    summary["overall_success"] = False

        # Calculate averages
        if throughputs:
            summary["avg_throughput"] = statistics.mean(throughputs)
            summary["max_throughput"] = max(throughputs)
            summary["min_throughput"] = min(throughputs)

        if ttfts:
            summary["avg_ttft_ms"] = statistics.mean(ttfts)

        if tpots:
            summary["avg_tpot_ms"] = statistics.mean(tpots)

        # Find best and worst tests
        if test_scores:
            summary["best_test"] = max(test_scores, key=test_scores.get)
            summary["worst_test"] = min(test_scores, key=test_scores.get)

        # Calculate overall score (0-100)
        if throughputs and summary["total_tests"] > 0:
            success_rate = summary["successful_tests"] / summary["total_tests"]
            throughput_score = min(100, summary["avg_throughput"] / 10)  # Normalize
            summary["overall_score"] = (success_rate * 50) + (throughput_score * 0.5)
        else:
            summary["overall_score"] = 0.0

        return summary


def save_benchmark_results(results: Dict[str, Any], output_path: Path):
    """Save benchmark results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_benchmark_results(path: Path) -> Optional[Dict[str, Any]]:
    """Load benchmark results from JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠  Failed to load benchmark results: {e}")
        return None


def compare_benchmarks(
    baseline: Dict[str, Any], candidate: Dict[str, Any]
) -> ComparisonResult:
    """
    Compare two benchmark results.

    Args:
        baseline: Baseline benchmark results
        candidate: Candidate benchmark results

    Returns:
        ComparisonResult with analysis
    """
    baseline_context = baseline["config"].get("max_model_len", 0)
    candidate_context = candidate["config"].get("max_model_len", 0)

    baseline_throughput = baseline["summary"].get("avg_throughput", 0)
    candidate_throughput = candidate["summary"].get("avg_throughput", 0)

    baseline_latency = baseline["summary"].get("avg_ttft_ms", 0)
    candidate_latency = candidate["summary"].get("avg_ttft_ms", 0)

    # Calculate differences
    if baseline_throughput > 0:
        throughput_diff_pct = (
            (candidate_throughput - baseline_throughput) / baseline_throughput
        ) * 100
    else:
        throughput_diff_pct = 0.0

    if baseline_latency > 0:
        latency_diff_pct = (
            (candidate_latency - baseline_latency) / baseline_latency
        ) * 100
    else:
        latency_diff_pct = 0.0

    # Determine winner
    if throughput_diff_pct > 5 and latency_diff_pct < 10:
        winner = "candidate"
        improvement_pct = throughput_diff_pct
        recommendation = f"Candidate is better: {throughput_diff_pct:+.1f}% throughput"
    elif throughput_diff_pct < -5:
        winner = "baseline"
        improvement_pct = throughput_diff_pct
        recommendation = f"Baseline is better: {throughput_diff_pct:+.1f}% throughput"
    else:
        winner = "tie"
        improvement_pct = 0.0
        recommendation = "No significant difference"

    return ComparisonResult(
        baseline_context=baseline_context,
        candidate_context=candidate_context,
        throughput_diff_pct=throughput_diff_pct,
        latency_diff_pct=latency_diff_pct,
        winner=winner,
        improvement_pct=improvement_pct,
        recommendation=recommendation,
    )
