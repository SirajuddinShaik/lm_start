# lm-start Project Guide

This document describes the lm-start deployment pipeline based on actual code implementation.

---

## Overview

lm-start automates vLLM model deployment through a structured 9-phase pipeline with agent-driven optimization.


## Phase 1: INIT

**Location**: `lm_start/phases/__init__.py:144` - `phase_init()`

**Purpose**: Initialize model directory structure and hardware detection

**Operations**:
1. Creates model directory structure:
   - `<model_dir>/` - Main model directory
   - `<model_dir>/logs/` - Log files
   - `<model_dir>/.llm-context/model-context/` - Context files

2. Hardware Detection via `HardwareDetector`:
   - Detects GPU count, names, memory
   - Detects CUDA version
   - Detects CPU count and RAM

3. Creates `device_config.json`:
```json
{
  "device_name": "Auto-Detected",
  "description": "8x NVIDIA H200",
  "cuda": {"version": "12.9", "home": "/usr/local/cuda"},
  "gpus": {
    "count": 8,
    "names": ["NVIDIA H200"],
    "memory_gb_per_gpu": 141.0,
    "total_memory_gb": 1128.0,
    "compute_capability": "9.0"
  },
  "system": {
    "total_ram_gb": 2015.0,
    "cpu_count": 96
  }
}
```

**State Management**:
- Uses `StateManager` to track phase completion
- Creates/resumes `setup_state.json`

