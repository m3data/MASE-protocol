#!/usr/bin/env python3
"""
Run remaining E001 pairs (2-5).
Each pair: 21 turns, matched single vs multi model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.experiment import ExperimentRunner
from src.ollama_client import OllamaClient

PROVOCATIONS = [
    {
        "id": "p002_children",
        "text": """If we gave children genuine decision-making power over issues
that affect their future, what would change? What are adults
protecting when they resist this?""",
        "seed": 43
    },
    {
        "id": "p003_emergency",
        "text": """Why don't we treat ongoing suffering with the same urgency
as acute emergencies? What would change if we did?""",
        "seed": 44
    },
    {
        "id": "p004_land",
        "text": """What does the land remember that we have forgotten?
How might we learn to listen again?""",
        "seed": 45
    },
    {
        "id": "p005_profit",
        "text": """Can profit-seeking ever be compatible with genuine care
for future generations? What would need to change for
markets to serve life rather than extract from it?""",
        "seed": 46
    }
]

def main():
    if not OllamaClient.is_running():
        print("Error: Ollama not running. Start with: ollama serve")
        sys.exit(1)

    print("=" * 60)
    print("E001: Running Remaining Pairs (2-5)")
    print("=" * 60)
    print(f"Pairs to run: {len(PROVOCATIONS)}")
    print("Turns per pair: 21")
    print("Estimated time: ~80 minutes total")
    print("=" * 60)

    runner = ExperimentRunner(
        single_model_config_path=Path("experiments/config/single_model.yaml"),
        multi_model_config_path=Path("experiments/config/multi_model.yaml"),
        output_dir=Path("experiments/runs"),
        agents_dir=Path(".claude/agents")
    )

    results = []
    for i, prov in enumerate(PROVOCATIONS, start=2):
        print(f"\n{'#' * 60}")
        print(f"# PAIR {i}/5: {prov['id']}")
        print(f"{'#' * 60}")

        result = runner.run_pair(
            provocation=prov["text"],
            seed=prov["seed"],
            provocation_id=prov["id"],
            max_turns=21
        )
        results.append(result)

        print(f"\nPair {i} complete: Δα = {result.delta_alpha:+.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("E001 PAIRS 2-5 COMPLETE")
    print("=" * 60)
    print(f"\n{'Pair':<6} {'Provocation':<20} {'Δκ':<10} {'Δα':<10} {'ΔH':<10}")
    print("-" * 56)
    for i, r in enumerate(results, start=2):
        print(f"{i:<6} {r.provocation_id:<20} {r.delta_curvature:+.4f}    {r.delta_alpha:+.4f}    {r.delta_entropy:+.4f}")

    # Aggregate
    import numpy as np
    mean_dk = np.mean([r.delta_curvature for r in results])
    mean_da = np.mean([r.delta_alpha for r in results])
    mean_dh = np.mean([r.delta_entropy for r in results])

    print("-" * 56)
    print(f"{'Mean':<6} {'':<20} {mean_dk:+.4f}    {mean_da:+.4f}    {mean_dh:+.4f}")

    return results

if __name__ == "__main__":
    main()
