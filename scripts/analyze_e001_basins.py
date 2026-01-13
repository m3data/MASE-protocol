#!/usr/bin/env python3
"""
Analyze E001 experiment pairs using basin detection.

Re-analyzes all E001 matched pairs with the new basin detection
to understand how model diversity affects dialogue quality beyond DFA alpha.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.session_analysis import analyze_session, compare_sessions


def main():
    runs_dir = Path("experiments/runs")

    # Find all pair directories
    pair_dirs = sorted([d for d in runs_dir.iterdir() if d.name.startswith("pair_")])

    print("E001 Basin Analysis")
    print("=" * 70)
    print(f"Found {len(pair_dirs)} matched pairs\n")

    # Collect results
    results = []

    for pair_dir in pair_dirs:
        single_dir = pair_dir / "single_model"
        multi_dir = pair_dir / "multi_model"

        # Find session files (non-checkpoint)
        single_sessions = list(single_dir.glob("session_*.json"))
        single_sessions = [s for s in single_sessions if "checkpoint" not in s.name]

        multi_sessions = list(multi_dir.glob("session_*.json"))
        multi_sessions = [s for s in multi_sessions if "checkpoint" not in s.name]

        if not single_sessions or not multi_sessions:
            print(f"Skipping {pair_dir.name}: missing sessions")
            continue

        single_path = single_sessions[0]
        multi_path = multi_sessions[0]

        # Load pair metadata
        pair_result_path = pair_dir / "pair_result.json"
        provocation_id = "unknown"
        if pair_result_path.exists():
            with open(pair_result_path) as f:
                pair_data = json.load(f)
                provocation_id = pair_data.get("provocation_id", "unknown")

        print(f"\n{pair_dir.name} ({provocation_id})")
        print("-" * 50)

        # Analyze both
        single_result = analyze_session(single_path)
        multi_result = analyze_session(multi_path)

        # Compute deltas (multi - single)
        delta_inquiry = multi_result.inquiry_vs_mimicry_ratio - single_result.inquiry_vs_mimicry_ratio
        delta_voice = multi_result.voice_distinctiveness - single_result.voice_distinctiveness
        delta_transitions = multi_result.transition_count - single_result.transition_count

        # Count locked patterns
        single_locked = single_result.coherence_pattern_distribution.get('locked', 0)
        multi_locked = multi_result.coherence_pattern_distribution.get('locked', 0)
        delta_locked = multi_locked - single_locked

        # Store result
        results.append({
            'pair': pair_dir.name,
            'provocation': provocation_id,
            'single_inquiry_ratio': single_result.inquiry_vs_mimicry_ratio,
            'multi_inquiry_ratio': multi_result.inquiry_vs_mimicry_ratio,
            'delta_inquiry': delta_inquiry,
            'single_voice_dist': single_result.voice_distinctiveness,
            'multi_voice_dist': multi_result.voice_distinctiveness,
            'delta_voice': delta_voice,
            'single_dominant': single_result.dominant_basin,
            'multi_dominant': multi_result.dominant_basin,
            'delta_transitions': delta_transitions,
            'delta_locked': delta_locked,
        })

        # Print comparison
        print(f"  {'Metric':<25} {'Single':<12} {'Multi':<12} {'Delta':<10}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        print(f"  {'Inquiry Ratio':<25} {single_result.inquiry_vs_mimicry_ratio:<12.3f} {multi_result.inquiry_vs_mimicry_ratio:<12.3f} {delta_inquiry:+.3f}")
        print(f"  {'Voice Distinctiveness':<25} {single_result.voice_distinctiveness:<12.3f} {multi_result.voice_distinctiveness:<12.3f} {delta_voice:+.3f}")
        print(f"  {'Transitions':<25} {single_result.transition_count:<12} {multi_result.transition_count:<12} {delta_transitions:+d}")
        print(f"  {'Locked Patterns':<25} {single_locked:<12} {multi_locked:<12} {delta_locked:+d}")
        print(f"  {'Dominant Basin':<25} {single_result.dominant_basin:<12} {multi_result.dominant_basin:<12}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("AGGREGATE ANALYSIS")
    print("=" * 70)

    if results:
        import numpy as np

        delta_inquiries = [r['delta_inquiry'] for r in results]
        delta_voices = [r['delta_voice'] for r in results]
        delta_lockeds = [r['delta_locked'] for r in results]

        print(f"\nInquiry Ratio (multi - single):")
        print(f"  Mean: {np.mean(delta_inquiries):+.3f}")
        print(f"  Std:  {np.std(delta_inquiries):.3f}")
        print(f"  Pairs favoring multi: {sum(1 for d in delta_inquiries if d > 0)}/{len(delta_inquiries)}")

        print(f"\nVoice Distinctiveness (multi - single):")
        print(f"  Mean: {np.mean(delta_voices):+.4f}")
        print(f"  Std:  {np.std(delta_voices):.4f}")
        print(f"  Pairs favoring multi: {sum(1 for d in delta_voices if d > 0)}/{len(delta_voices)}")

        print(f"\nLocked Patterns (multi - single):")
        print(f"  Mean: {np.mean(delta_lockeds):+.1f}")
        print(f"  Pairs with fewer locked in multi: {sum(1 for d in delta_lockeds if d < 0)}/{len(delta_lockeds)}")

        # Provocation breakdown
        print(f"\nBy Provocation:")
        for r in results:
            emoji = "+" if r['delta_inquiry'] > 0 else "-"
            print(f"  {r['provocation']:<20} inquiry: {r['delta_inquiry']:+.3f} {emoji}")

    # Save results
    output_path = Path("experiments/analysis/E001_model_diversity/basin_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
