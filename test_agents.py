#!/usr/bin/env python3
"""
Test script for lm-start agents.

Usage:
    python test_agents.py phase <model_dir> <phase_name> [error_message]
    python test_agents.py summarizer <model_dir> <run_folder>
    python test_agents.py planner <model_dir>

Examples:
    python test_agents.py phase /data/models/moonshotai-Kimi-K2.6 venv "pip install failed"
    python test_agents.py summarizer /data/models/moonshotai-Kimi-K2.6 run_1_1
    python test_agents.py planner /data/models/moonshotai-Kimi-K2.6
"""

import sys
import os
from pathlib import Path

# Add lm_start to path
sys.path.insert(0, str(Path(__file__).parent))

from lm_start.scripts.opencode_phase_agent import OpenCodePhaseAgent
from lm_start.scripts.orchestrator import OpenCodeAgenticOrchestrator


def test_phase_agent(model_dir: str, phase: str, error: str = "Test error"):
    """Test the phase recovery agent."""
    print(f"\n{'=' * 60}")
    print(f"TESTING PHASE AGENT")
    print(f"{'=' * 60}")
    print(f"Model Dir: {model_dir}")
    print(f"Phase: {phase}")
    print(f"Error: {error}")
    print(f"{'=' * 60}\n")

    agent = OpenCodePhaseAgent(model_dir=model_dir, phase=phase, error=error)

    print("Running phase recovery...")
    result = agent.run()

    print(f"\n{'=' * 60}")
    print(f"RESULT:")
    print(f"{'=' * 60}")
    print(f"Success: {result.get('success', False)}")
    print(f"Attempts: {result.get('attempts', 0)}")
    print(f"Session ID: {result.get('session_id', 'N/A')}")

    if result.get("fixes"):
        print(f"\nFixes applied:")
        for fix in result["fixes"]:
            print(f"  - {fix}")

    if result.get("error"):
        print(f"\nError: {result['error']}")

    return result


def test_summarizer_agent(model_dir: str, run_folder: str):
    """Test the summarizer agent."""
    print(f"\n{'=' * 60}")
    print(f"TESTING SUMMARIZER AGENT")
    print(f"{'=' * 60}")
    print(f"Model Dir: {model_dir}")
    print(f"Run Folder: {run_folder}")
    print(f"{'=' * 60}\n")

    model_path = Path(model_dir)
    run_path = model_path / ".runs" / run_folder

    if not run_path.exists():
        print(f"❌ Error: Run folder not found: {run_path}")
        return None

    # Load result.json
    result_file = run_path / "result.json"
    if not result_file.exists():
        print(f"❌ Error: result.json not found in {run_path}")
        return None

    import json

    with open(result_file) as f:
        result = json.load(f)

    print(f"Loaded result from {result_file}")
    print(f"Success: {result.get('success', False)}")

    # Create orchestrator and spawn summarizer
    print("\nSpawning summarizer agent...")
    orchestrator = OpenCodeAgenticOrchestrator(
        model_dir=model_dir, model_id="test-model"
    )

    summary_result = orchestrator.spawn_summarizer_agent(run_folder, result)

    print(f"\n{'=' * 60}")
    print(f"SUMMARIZER RESULT:")
    print(f"{'=' * 60}")
    print(f"Success: {summary_result.get('success', False)}")
    print(f"Session ID: {summary_result.get('session_id', 'N/A')}")

    # Check if summary.json was created
    summary_file = run_path / "summary.json"
    if summary_file.exists():
        print(f"✅ Summary file created: {summary_file}")
        with open(summary_file) as f:
            summary_data = json.load(f)
        print(f"\nSummary preview:")
        print(f"  Status: {summary_data.get('status', 'N/A')}")
        print(
            f"  Throughput: {summary_data.get('performance', {}).get('throughput_tok_s', 'N/A')} tok/s"
        )
    else:
        print(f"⚠️  Summary file not yet created (may need to check session)")

    return summary_result


def test_planner_agent(model_dir: str):
    """Test the planner agent."""
    print(f"\n{'=' * 60}")
    print(f"TESTING PLANNER AGENT")
    print(f"{'=' * 60}")
    print(f"Model Dir: {model_dir}")
    print(f"{'=' * 60}\n")

    print("Creating orchestrator...")
    orchestrator = OpenCodeAgenticOrchestrator(
        model_dir=model_dir,
        model_id="test-model",
        goal="test configuration optimization",
    )

    print("Setting up directories...")
    orchestrator.setup_directories()

    print("\nSpawning planner agent (this may take a few minutes)...")
    plan = orchestrator.spawn_planner_agent()

    print(f"\n{'=' * 60}")
    print(f"PLANNER RESULT:")
    print(f"{'=' * 60}")

    if plan.get("experiments_queued"):
        print(f"✅ Plan generated successfully")
        print(f"Experiments queued: {len(plan['experiments_queued'])}")

        for i, exp in enumerate(plan["experiments_queued"], 1):
            print(f"\n  Experiment {i}:")
            print(f"    ID: {exp.get('id', 'N/A')}")
            print(f"    Name: {exp.get('name', 'N/A')}")
            print(f"    Hypothesis: {exp.get('hypothesis', 'N/A')[:60]}...")

        # Show insights
        if plan.get("insights_read"):
            print(f"\n  Insights captured: ✅")
            print(f"    Length: {len(plan['insights_read'])} chars")

        if plan.get("knowledge_update"):
            print(f"\n  Knowledge update captured: ✅")
            print(f"    Length: {len(plan['knowledge_update'])} chars")
    else:
        print(f"⚠️  No experiments queued")
        if plan.get("optimization_complete"):
            print(f"    Reason: Optimization marked as complete")
        if plan.get("error"):
            print(f"    Error: {plan['error']}")

    return plan


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    agent_type = sys.argv[1]
    model_dir = sys.argv[2]

    if agent_type == "phase":
        if len(sys.argv) < 4:
            print("Error: phase agent requires phase name")
            print(
                "Usage: python test_agents.py phase <model_dir> <phase_name> [error_message]"
            )
            sys.exit(1)

        phase = sys.argv[3]
        error = sys.argv[4] if len(sys.argv) > 4 else "Test error for phase recovery"
        result = test_phase_agent(model_dir, phase, error)

    elif agent_type == "summarizer":
        if len(sys.argv) < 4:
            print("Error: summarizer agent requires run_folder")
            print("Usage: python test_agents.py summarizer <model_dir> <run_folder>")
            sys.exit(1)

        run_folder = sys.argv[3]
        result = test_summarizer_agent(model_dir, run_folder)

    elif agent_type == "planner":
        result = test_planner_agent(model_dir)

    else:
        print(f"❌ Unknown agent type: {agent_type}")
        print("Valid types: phase, summarizer, planner")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"TEST COMPLETE")
    print(f"{'=' * 60}\n")

    return result


if __name__ == "__main__":
    main()
