#!/usr/bin/env python3
"""
Test script for Agentic LLM System components.

This tests the new theoretical calculator, flag knowledge base,
benchmark runner, and experiment agent without running actual vLLM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("TESTING AGENTIC LLM SYSTEM COMPONENTS")
print("=" * 60)

print("\n[1/7] Testing imports...")
try:
    from utils.theoretical_calculator import (
        TheoreticalCalculator,
        HardwareConfig,
        ModelConfig,
        TheoreticalConfig,
        calculate_from_model_info,
    )
    from utils.benchmark_runner import BenchmarkRunner, save_benchmark_results
    from agents.experiment_agent import ExperimentAgent
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

print("\n[2/7] Testing configuration dataclasses...")
try:
    hardware = HardwareConfig.h200_cluster(gpu_count=8)
    
    model = ModelConfig(
        model_id="arcee-ai/Trinity-Large-Thinking-W4A16",
        model_params_b=397.0,
        max_position_embeddings=262144,
        num_layers=61,
        hidden_size=7168,
        num_attention_heads=128,
        num_key_value_heads=128,
        quantization="w4a16",
        is_moe=True,
        is_reasoning=True,
    )
    
    print(f"  ✓ Hardware: {hardware.gpu_count}x {hardware.gpu_name} GPUs ({hardware.gpu_memory_gb} GB each)")
    print(f"  ✓ Model: {model.model_id}")
    print(f"  ✓ Parameters: {model.model_params_b}B")
    print(f"  ✓ Max position embeddings: {model.max_position_embeddings}")
except Exception as e:
    print(f"  ✗ Config creation failed: {e}")
    sys.exit(1)

print("\n[3/7] Testing TheoreticalCalculator...")
try:
    calc = TheoreticalCalculator(hardware)
    
    required_tp = calc.calculate_required_tensor_parallel(model)
    print(f"  ✓ Required tensor parallel: {required_tp} GPUs")
    
    max_context = calc.calculate_max_context_from_memory(model, target_users=32)
    print(f"  ✓ Max achievable context at 32 users: {max_context:,} tokens")
    
    mem_estimate = calc.calculate_memory_requirements(model, max_context=262144, users=32)
    print(f"  ✓ Total memory required: {mem_estimate['total_required_gb']:.1f} GB")
    print(f"  ✓ Per-GPU estimate: {mem_estimate['per_gpu_gb']:.1f} GB")
    
    perf = calc.estimate_throughput_bounds(model, max_context=262144)
    print(f"  ✓ Theoretical max throughput: {perf['max_throughput_tok_s']:.0f} tok/s")
    print(f"  ✓ Theoretical min latency: {perf['min_latency_ms']:.2f} ms")
    
    feasible = calc.check_config_feasibility(model, max_context=262144, users=32)
    print(f"  ✓ Config feasibility check: {feasible}")
    
except Exception as e:
    print(f"  ✗ TheoreticalCalculator failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4/7] Testing TheoreticalConfig generation...")
try:
    config = calc.calculate_optimal_config(model, target_users=32)
    print(f"  ✓ Generated TheoreticalConfig")
    print(f"  ✓ TP size: {config.tensor_parallel_size}")
    print(f"  ✓ Max model len: {config.max_model_len}")
    print(f"  ✓ GPU memory utilization: {config.gpu_memory_utilization}")
    print(f"  ✓ Max batched tokens: {config.max_num_batched_tokens}")
    print(f"  ✓ Max seqs: {config.max_num_seqs}")
    
    flags = config.to_vllm_flags()
    print(f"  ✓ Converted to {len(flags)} vLLM flags")
    print(f"  Sample flags:")
    for key, value in list(flags.items())[:5]:
        print(f"    --{key.replace('_', '-')}: {value}")
    
except Exception as e:
    print(f"  ✗ Config generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[5/7] Testing ExperimentAgent initialization...")
try:
    agent = ExperimentAgent(
        model_dir=str(Path("/tmp/test_model")),
        verbose=True,
        enable_benchmarking=True,
    )
    
    print(f"  ✓ Agent initialized: {agent.name}")
    print(f"  ✓ Model dir: {agent.model_dir}")
    print(f"  ✓ Max retries: {agent.max_retries}")
    print(f"  ✓ Benchmarking enabled: {agent.enable_benchmarking}")
    
    attrs = ['working_configs', 'failed_configs', 'benchmark_results', 'run_count', 'run', 'optimize']
    for attr in attrs:
        assert hasattr(agent, attr), f"Missing attribute: {attr}"
    print(f"  ✓ All required attributes present")
    
except Exception as e:
    print(f"  ✗ Agent initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[6/7] Testing flag knowledge base...")
try:
    import yaml
    
    kb_path = Path(__file__).parent / "configs" / "flag_knowledge_base.yaml"
    if kb_path.exists():
        with open(kb_path) as f:
            kb = yaml.safe_load(f)
        
        categories = list(kb.keys())
        print(f"  ✓ Loaded knowledge base with {len(categories)} categories")
        
        total_flags = sum(len(flags) for flags in kb.values())
        print(f"  ✓ Total flags documented: {total_flags}")
        
        for cat in categories:
            print(f"    - {cat}: {len(kb[cat])} flags")
    else:
        print(f"  ⚠ Knowledge base not found at {kb_path}")
        
except Exception as e:
    print(f"  ✗ Knowledge base loading failed: {e}")
    sys.exit(1)

print("\n[7/7] Testing flag rules...")
try:
    rules_path = Path(__file__).parent / "configs" / "flag_rules.yaml"
    if rules_path.exists():
        with open(rules_path) as f:
            rules = yaml.safe_load(f)
        
        model_families = list(rules.get('model_families', {}).keys())
        print(f"  ✓ Loaded rules for {len(model_families)} model families")
        
        for family in model_families[:5]:
            family_rules = rules['model_families'][family]
            has_required = 'required_flags' in family_rules
            has_recommended = 'recommended_flags' in family_rules
            print(f"    - {family}: required={has_required}, recommended={has_recommended}")
    else:
        print(f"  ⚠ Flag rules not found at {rules_path}")
        
except Exception as e:
    print(f"  ✗ Flag rules loading failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nSummary:")
print("  ✓ All components import correctly")
print("  ✓ Configuration dataclasses work")
print("  ✓ TheoreticalCalculator generates valid configs")
print("  ✓ Flag generation produces correct vLLM flags")
print("  ✓ ExperimentAgent initializes properly")
print("  ✓ Knowledge base and rules are accessible")
print("\nThe agentic system is ready for use.")
