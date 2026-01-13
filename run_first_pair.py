#!/usr/bin/env python3
"""
Run first matched-pair experiment for MASE.
Quick validation: 7 turns on provocation p001_success.
"""

import sys
from pathlib import Path

# Add parent to path so we can import src as a package
sys.path.insert(0, str(Path(__file__).parent))

from src.experiment import ExperimentRunner
from src.ollama_client import OllamaClient

def main():
    # Check Ollama
    if not OllamaClient.is_running():
        print("Error: Ollama not running. Start with: ollama serve")
        sys.exit(1)

    print("MASE First Matched-Pair Experiment")
    print("=" * 50)
    print("Provocation: p001_success (What does success mean...)")
    print("Turns: 21 (full experiment)")
    print("Seed: 42")
    print("=" * 50)

    # Set up runner
    runner = ExperimentRunner(
        single_model_config_path=Path("experiments/config/single_model.yaml"),
        multi_model_config_path=Path("experiments/config/multi_model.yaml"),
        output_dir=Path("experiments/runs"),
        agents_dir=Path(".claude/agents")
    )

    # Provocation from p001_success
    provocation = """What does success mean when we're trying to coordinate action
on problems that span generations? How do we measure progress
on things that unfold in geological time?"""

    # Run the pair
    result = runner.run_pair(
        provocation=provocation,
        seed=42,
        provocation_id="p001_success",
        max_turns=21
    )

    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETE")
    print("=" * 50)
    print(f"Pair ID: {result.pair_id}")
    print(f"Results saved to: experiments/runs/pair_{result.pair_id}/")

    return result

if __name__ == "__main__":
    main()
