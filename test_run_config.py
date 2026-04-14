#!/usr/bin/env python3
"""Test a specific run config with 4 benchmark variations."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from utils.benchmark_runner import BenchmarkRunner, save_benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Test a run config with benchmarks")
    parser.add_argument(
        "run_dir", help="Path to run directory (e.g., /data/models/model/.runs/run_1)"
    )
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_file = run_dir / "config.json"

    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    print("=" * 70)
    print(f"TESTING CONFIG FROM: {run_dir}")
    print("=" * 70)
    print(f"max_model_len: {config.get('max_model_len')}")
    print(f"tensor_parallel_size: {config.get('tensor_parallel_size')}")
    print(f"dtype: {config.get('dtype')}")
    print(f"port: {args.port}")
    print()

    model_dir = run_dir.parent.parent
    venv_python = model_dir / ".venv" / "bin" / "python"

    model_id = config.get("model_id") or config.get("model")
    model_info_path = (
        run_dir.parent.parent / ".llm-context" / "model-context" / "model_info.json"
    )
    if not model_id and model_info_path.exists():
        with open(model_info_path) as f:
            model_info = json.load(f)
            model_id = model_info.get("model_id")
    if not model_id:
        model_id = "moonshotai/Kimi-K2.5"

    runner = BenchmarkRunner(
        model_id=model_id,
        vllm_port=args.port,
        venv_python=venv_python,
        verbose=True,
    )

    max_context = config.get("max_model_len", 32768)
    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "tests": {},
    }

    tests = [
        {
            "name": "small",
            "input_len": min(512, max_context),
            "output_len": 100,
            "num_prompts": 10,
            "max_concurrency": 2,
        },
        {
            "name": "medium",
            "input_len": min(4096, max_context),
            "output_len": 200,
            "num_prompts": 20,
            "max_concurrency": 4,
        },
        {
            "name": "large",
            "input_len": min(8192, max_context),
            "output_len": 500,
            "num_prompts": 30,
            "max_concurrency": 8,
        },
        {
            "name": "limit",
            "input_len": max_context,
            "output_len": min(1000, max_context // 10),
            "num_prompts": 40,
            "max_concurrency": 16,
        },
    ]

    print("Running 4 benchmark tests...")
    print()

    for test in tests:
        print(f"\n{'=' * 70}")
        print(f"TEST: {test['name'].upper()}")
        print(f"  Input: {test['input_len']} tokens")
        print(f"  Output: {test['output_len']} tokens")
        print(f"  Prompts: {test['num_prompts']}")
        print(f"  Concurrency: {test['max_concurrency']}")
        print(f"{'=' * 70}")

        result = runner.run_serve_benchmark(
            config=config,
            input_len=test["input_len"],
            output_len=test["output_len"],
            test_name=test["name"],
            num_prompts=test["num_prompts"],
            max_concurrency=test["max_concurrency"],
            timeout=300,
        )

        results["tests"][test["name"]] = {
            "success": result.success,
            "metrics": {
                "throughput_tok_s": result.metrics.throughput_tok_s,
                "ttft_ms": result.metrics.ttft_ms,
                "tpot_ms": result.metrics.tpot_ms,
                "successful_requests": result.metrics.successful_requests,
                "failed_requests": result.metrics.failed_requests,
                "duration_seconds": result.metrics.duration_seconds,
            },
            "error": result.error,
        }

        print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
        if result.success:
            print(f"  Throughput: {result.metrics.throughput_tok_s:.2f} tok/s")
            print(f"  TTFT: {result.metrics.ttft_ms:.2f} ms")
            print(f"  TPOT: {result.metrics.tpot_ms:.2f} ms")
        else:
            print(f"  Error: {result.error}")

    output_file = run_dir / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"RESULTS SAVED TO: {output_file}")
    print(f"{'=' * 70}")

    print("\nSUMMARY:")
    for name, test_result in results["tests"].items():
        status = "✓" if test_result["success"] else "✗"
        print(f"  {status} {name}: ", end="")
        if test_result["success"]:
            print(f"{test_result['metrics']['throughput_tok_s']:.1f} tok/s")
        else:
            print(f"FAILED - {test_result['error']}")


if __name__ == "__main__":
    main()
