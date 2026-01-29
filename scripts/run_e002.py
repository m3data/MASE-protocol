#!/usr/bin/env python3
"""
Run E002: Personality Effect on Basin Distribution experiment.

Compares personality-enabled sessions with base-temperature control.

Usage:
    python scripts/run_e002.py --pair 1
    python scripts/run_e002.py --all
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import EnsembleConfig
from src.orchestrator import DialogueOrchestrator
from src.session_analysis import analyze_session, compare_sessions


# E002 configuration
PROVOCATIONS = {
    'p001_success': """What does success mean when we're trying to coordinate action
on problems that span generations? How do we measure progress
on things that unfold in geological time?""",

    'p004_land': """What does the land remember that we have forgotten?
How might we learn to listen again?""",

    'p005_profit': """Can profit-seeking ever be compatible with genuine care
for future generations? What would need to change for
markets to serve life rather than extract from it?""",
}

# Seeds for replication
SEEDS = [100, 101]


def run_pair(
    provocation_id: str,
    seed: int,
    output_dir: Path,
    agents_dir: Path
) -> dict:
    """
    Run a matched pair (personality vs no-personality).

    Args:
        provocation_id: ID of provocation to use
        seed: Random seed
        output_dir: Directory for output
        agents_dir: Path to agent definitions

    Returns:
        dict with pair results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pair_dir = output_dir / f"e002_pair_{timestamp}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    provocation = PROVOCATIONS[provocation_id]

    print(f"\n{'='*60}")
    print(f"E002 Pair: {provocation_id} (seed={seed})")
    print(f"{'='*60}")

    # --- Personality-enabled condition ---
    print(f"\n[1/2] Running personality-enabled condition...")
    personality_dir = pair_dir / "personality"
    personality_dir.mkdir(exist_ok=True)

    config_path = Path("experiments/config/multi_model.yaml")
    config = EnsembleConfig.from_yaml(config_path)

    orchestrator = DialogueOrchestrator(config, agents_dir=agents_dir)
    personality_session = orchestrator.run_dialogue(
        provocation=provocation,
        output_dir=personality_dir,
        seed=seed,
        provocation_id=provocation_id,
        config_path=str(config_path)
    )

    # --- No-personality condition ---
    print(f"\n[2/2] Running no-personality condition...")
    control_dir = pair_dir / "no_personality"
    control_dir.mkdir(exist_ok=True)

    config_path = Path("experiments/config/multi_model_no_personality.yaml")
    config = EnsembleConfig.from_yaml(config_path)

    orchestrator = DialogueOrchestrator(config, agents_dir=agents_dir)
    control_session = orchestrator.run_dialogue(
        provocation=provocation,
        output_dir=control_dir,
        seed=seed,
        provocation_id=provocation_id,
        config_path=str(config_path)
    )

    # --- Analyze both ---
    print(f"\nAnalyzing sessions...")
    personality_result = analyze_session(personality_session)
    control_result = analyze_session(control_session)

    # Compute deltas (personality - control)
    delta_inquiry = personality_result.inquiry_vs_mimicry_ratio - control_result.inquiry_vs_mimicry_ratio
    delta_voice = personality_result.voice_distinctiveness - control_result.voice_distinctiveness

    control_locked = control_result.coherence_pattern_distribution.get('locked', 0)
    personality_locked = personality_result.coherence_pattern_distribution.get('locked', 0)
    delta_locked = personality_locked - control_locked

    pair_result = {
        'timestamp': timestamp,
        'provocation_id': provocation_id,
        'seed': seed,
        'personality_session': str(personality_session),
        'control_session': str(control_session),

        # Personality condition metrics
        'personality_inquiry_ratio': personality_result.inquiry_vs_mimicry_ratio,
        'personality_voice_dist': personality_result.voice_distinctiveness,
        'personality_dominant_basin': personality_result.dominant_basin,
        'personality_locked': personality_locked,
        'personality_alpha': personality_result.dfa_alpha,

        # Control condition metrics
        'control_inquiry_ratio': control_result.inquiry_vs_mimicry_ratio,
        'control_voice_dist': control_result.voice_distinctiveness,
        'control_dominant_basin': control_result.dominant_basin,
        'control_locked': control_locked,
        'control_alpha': control_result.dfa_alpha,

        # Deltas
        'delta_inquiry': delta_inquiry,
        'delta_voice': delta_voice,
        'delta_locked': delta_locked,
        'delta_alpha': personality_result.dfa_alpha - control_result.dfa_alpha,
    }

    # Save pair result
    result_path = pair_dir / "pair_result.json"
    with open(result_path, 'w') as f:
        json.dump(pair_result, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Pair Summary: {provocation_id} (seed={seed})")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Personality':<12} {'Control':<12} {'Delta':<10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Inquiry Ratio':<20} {personality_result.inquiry_vs_mimicry_ratio:<12.3f} {control_result.inquiry_vs_mimicry_ratio:<12.3f} {delta_inquiry:+.3f}")
    print(f"  {'Voice Dist':<20} {personality_result.voice_distinctiveness:<12.4f} {control_result.voice_distinctiveness:<12.4f} {delta_voice:+.4f}")
    print(f"  {'Locked':<20} {personality_locked:<12} {control_locked:<12} {delta_locked:+d}")
    print(f"  {'Dominant Basin':<20} {personality_result.dominant_basin:<12} {control_result.dominant_basin:<12}")

    return pair_result


def main():
    parser = argparse.ArgumentParser(description="Run E002 experiment")
    parser.add_argument('--pair', type=int, help="Run specific pair (1-6)")
    parser.add_argument('--all', action='store_true', help="Run all pairs")
    parser.add_argument('--dry-run', action='store_true', help="Print plan without running")
    args = parser.parse_args()

    # Setup paths
    output_dir = Path("experiments/runs/e002")
    agents_dir = Path("agents/personas")

    # Generate pair configurations
    pairs = []
    for prov_id in PROVOCATIONS.keys():
        for seed in SEEDS:
            pairs.append((prov_id, seed))

    print("E002: Personality Effect on Basin Distribution")
    print("=" * 60)
    print(f"Pairs to run: {len(pairs)}")
    for i, (prov_id, seed) in enumerate(pairs, 1):
        print(f"  {i}. {prov_id} (seed={seed})")

    if args.dry_run:
        print("\nDry run - no sessions will be executed.")
        return

    # Determine which pairs to run
    if args.pair:
        if args.pair < 1 or args.pair > len(pairs):
            print(f"Error: --pair must be between 1 and {len(pairs)}")
            return
        pairs_to_run = [pairs[args.pair - 1]]
    elif args.all:
        pairs_to_run = pairs
    else:
        print("\nSpecify --pair N or --all to run experiment.")
        return

    # Run selected pairs
    results = []
    for prov_id, seed in pairs_to_run:
        result = run_pair(prov_id, seed, output_dir, agents_dir)
        results.append(result)

    # Save aggregate results
    if len(results) > 1:
        import numpy as np

        aggregate_path = output_dir / "e002_aggregate.json"
        with open(aggregate_path, 'w') as f:
            json.dump({
                'pairs': results,
                'summary': {
                    'n_pairs': len(results),
                    'mean_delta_inquiry': float(np.mean([r['delta_inquiry'] for r in results])),
                    'mean_delta_voice': float(np.mean([r['delta_voice'] for r in results])),
                    'pairs_favoring_personality': sum(1 for r in results if r['delta_inquiry'] > 0),
                }
            }, f, indent=2)

        print(f"\n{'='*60}")
        print("E002 AGGREGATE RESULTS")
        print("=" * 60)
        print(f"Pairs completed: {len(results)}")
        print(f"Mean Δ(inquiry ratio): {np.mean([r['delta_inquiry'] for r in results]):+.3f}")
        print(f"Mean Δ(voice dist): {np.mean([r['delta_voice'] for r in results]):+.4f}")
        print(f"Pairs favoring personality: {sum(1 for r in results if r['delta_inquiry'] > 0)}/{len(results)}")


if __name__ == "__main__":
    main()
