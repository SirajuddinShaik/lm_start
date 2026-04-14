#!/usr/bin/env python3
"""
Test script for Experiment Agent V2 components.

This tests the new intelligent optimizer without running actual vLLM.
"""

import sys
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from utils.capacity_calculator import CapacityCalculator, calculate_capacity
    from utils.intelligent_config import IntelligentConfigGenerator, generate_config
    from utils.adaptive_validator import AdaptiveValidator
    from agents.experiment_agent_v2 import ExperimentAgentV2
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test capacity calculator
print("\n" + "=" * 60)
print("TEST 1: Capacity Calculator")
print("=" * 60)

calc = CapacityCalculator()
estimate = calc.calculate(
    model_params=57e9,
    num_layers=61,
    hidden_size=7168,
    max_context=262144,
    quantization='w4a16',
    gpu_count=8,
    gpu_memory_gb=141,
    gpu_type='H200',
    is_moe=True,
)

print(f"\nResults for Trinity (57B, W4A16, 8x H200):")
print(f"  Max concurrent users: {estimate.max_concurrent_users}")
print(f"  Max model len: {estimate.max_model_len:,}")
print(f"  KV cache per token: {estimate.kv_cache_per_token_bytes / 1024:.1f} KB")
print(f"  Total KV capacity: {estimate.total_kv_capacity_tokens:,} tokens")
print(f"  Bottleneck: {estimate.bottleneck}")

# Test config generator
print("\n" + "=" * 60)
print("TEST 2: Intelligent Config Generator")
print("=" * 60)

gen = IntelligentConfigGenerator()
config = gen.generate(estimate)

print(f"\nGenerated config:")
for k, v in config.items():
    if isinstance(v, int) and v > 1000:
        print(f"  {k}: {v:,}")
    else:
        print(f"  {k}: {v}")

# Test variant generation
print("\n" + "=" * 60)
print("TEST 3: Variant Generation")
print("=" * 60)

variants = gen.create_variants(config, estimate)

for name, v in variants.items():
    print(f"\n{name.upper()}:")
    print(f"  max_num_batched_tokens: {v['max_num_batched_tokens']:,}")
    print(f"  enable_chunked_prefill: {v['enable_chunked_prefill']}")

# Test with different model sizes
print("\n" + "=" * 60)
print("TEST 4: Different Model Sizes")
print("=" * 60)

test_models = [
    ("Llama-3-8B", 8e9, 32, 4096, "fp16", 8, 80, "A100"),
    ("Llama-3-70B", 70e9, 80, 8192, "fp16", 8, 80, "A100"),
    ("Mixtral-8x7B", 47e9, 32, 4096, "fp16", 8, 80, "A100"),
]

for name, params, layers, hidden, quant, gpus, mem, gpu_type in test_models:
    est = calc.calculate(
        model_params=params,
        num_layers=layers,
        hidden_size=hidden,
        max_context=32768,
        quantization=quant,
        gpu_count=gpus,
        gpu_memory_gb=mem,
        gpu_type=gpu_type,
        is_moe="mixtral" in name.lower(),
    )
    print(f"\n{name}:")
    print(f"  Users at 60% context: {est.max_concurrent_users}")
    print(f"  KV capacity: {est.total_kv_capacity_tokens / 1e6:.1f}M tokens")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
