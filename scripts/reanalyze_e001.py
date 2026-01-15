#!/usr/bin/env python3
"""
Reanalyze E001 sessions with trajectory and integrity metrics.

Adds new analysis dimensions:
- Trajectory: path_length, displacement, tortuosity, mean_velocity
- Integrity: score, label, autocorrelation, recurrence_rate
- Transformation density
"""

import sys
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from session_analysis import analyze_session

# E001 pairs from results.json
E001_PAIRS = [
    {"pair_id": "20260113_134345", "provocation": "p001_success"},
    {"pair_id": "20260113_153812", "provocation": "p002_children"},
    {"pair_id": "20260113_160103", "provocation": "p003_emergency"},
    {"pair_id": "20260113_161842", "provocation": "p004_land"},
    {"pair_id": "20260113_164140", "provocation": "p005_profit"},
]

def analyze_pair(pair_id: str, provocation: str) -> dict:
    """Analyze a single pair with new metrics."""
    base_path = Path(__file__).parent.parent / "experiments" / "runs" / f"pair_{pair_id}"

    # Find session files
    single_dir = base_path / "single_model"
    multi_dir = base_path / "multi_model"

    single_sessions = list(single_dir.glob("session_*.json"))
    multi_sessions = list(multi_dir.glob("session_*.json"))

    # Filter out checkpoints
    single_sessions = [s for s in single_sessions if "checkpoint" not in s.name]
    multi_sessions = [s for s in multi_sessions if "checkpoint" not in s.name]

    if not single_sessions or not multi_sessions:
        print(f"  Warning: Missing sessions for pair {pair_id}")
        return None

    # Analyze both
    print(f"\n  Analyzing single-model: {single_sessions[0].name}")
    single_result = analyze_session(single_sessions[0])

    print(f"  Analyzing multi-model: {multi_sessions[0].name}")
    multi_result = analyze_session(multi_sessions[0])

    return {
        "pair_id": pair_id,
        "provocation": provocation,
        "single": {
            # Original metrics
            "curvature": single_result.semantic_curvature,
            "alpha": single_result.dfa_alpha,
            "entropy": single_result.entropy_shift,
            # Trajectory metrics
            "path_length": single_result.trajectory_path_length,
            "displacement": single_result.trajectory_displacement,
            "tortuosity": single_result.trajectory_tortuosity,
            "mean_velocity": single_result.trajectory_mean_velocity,
            # Integrity metrics
            "integrity_score": single_result.integrity_score,
            "integrity_label": single_result.integrity_label,
            "autocorrelation": single_result.integrity_autocorrelation,
            "recurrence_rate": single_result.integrity_recurrence_rate,
            "transformation_density": single_result.transformation_density,
            # Quality
            "voice_distinctiveness": single_result.voice_distinctiveness,
            "inquiry_ratio": single_result.inquiry_vs_mimicry_ratio,
        },
        "multi": {
            # Original metrics
            "curvature": multi_result.semantic_curvature,
            "alpha": multi_result.dfa_alpha,
            "entropy": multi_result.entropy_shift,
            # Trajectory metrics
            "path_length": multi_result.trajectory_path_length,
            "displacement": multi_result.trajectory_displacement,
            "tortuosity": multi_result.trajectory_tortuosity,
            "mean_velocity": multi_result.trajectory_mean_velocity,
            # Integrity metrics
            "integrity_score": multi_result.integrity_score,
            "integrity_label": multi_result.integrity_label,
            "autocorrelation": multi_result.integrity_autocorrelation,
            "recurrence_rate": multi_result.integrity_recurrence_rate,
            "transformation_density": multi_result.transformation_density,
            # Quality
            "voice_distinctiveness": multi_result.voice_distinctiveness,
            "inquiry_ratio": multi_result.inquiry_vs_mimicry_ratio,
        }
    }


def compute_deltas(results: list) -> dict:
    """Compute deltas (multi - single) for each metric."""
    metrics = [
        "curvature", "alpha", "entropy",
        "path_length", "displacement", "tortuosity", "mean_velocity",
        "integrity_score", "autocorrelation", "recurrence_rate", "transformation_density",
        "voice_distinctiveness", "inquiry_ratio"
    ]

    deltas = {m: [] for m in metrics}

    for r in results:
        if r is None:
            continue
        for m in metrics:
            single_val = r["single"].get(m)
            multi_val = r["multi"].get(m)
            if single_val is not None and multi_val is not None:
                deltas[m].append(multi_val - single_val)

    return deltas


def print_comparison_table(results: list):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("E001 REANALYSIS WITH TRAJECTORY + INTEGRITY METRICS")
    print("=" * 80)

    # Per-pair results
    print("\n--- Per-Pair Results ---\n")

    header = f"{'Pair':<12} {'Prov':<10} | {'Δα':<8} {'Δint':<8} {'Δtort':<8} {'Δvel':<8} | {'S-int':<8} {'M-int':<8}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r is None:
            continue

        delta_alpha = r["multi"]["alpha"] - r["single"]["alpha"]
        delta_integrity = r["multi"]["integrity_score"] - r["single"]["integrity_score"]
        delta_tortuosity = (r["multi"]["tortuosity"] or 0) - (r["single"]["tortuosity"] or 0)
        delta_velocity = (r["multi"]["mean_velocity"] or 0) - (r["single"]["mean_velocity"] or 0)

        s_int = r["single"]["integrity_score"]
        m_int = r["multi"]["integrity_score"]

        print(f"{r['pair_id']:<12} {r['provocation']:<10} | {delta_alpha:>+7.3f} {delta_integrity:>+7.3f} {delta_tortuosity:>+7.3f} {delta_velocity:>+7.3f} | {s_int:>7.3f} {m_int:>7.3f}")

    # Aggregate deltas
    deltas = compute_deltas(results)

    print("\n--- Aggregate Deltas (Multi - Single) ---\n")

    for metric in ["alpha", "integrity_score", "tortuosity", "mean_velocity",
                   "path_length", "transformation_density", "voice_distinctiveness"]:
        vals = deltas[metric]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            # Simple t-test
            if std > 0 and len(vals) > 1:
                t = mean / (std / np.sqrt(len(vals)))
                from scipy import stats
                p = 2 * (1 - stats.t.cdf(abs(t), len(vals) - 1))
            else:
                t, p = 0, 1.0

            sig = "*" if p < 0.05 else ""
            print(f"  Δ{metric:<22}: {mean:>+7.3f} ± {std:>6.3f}  (t={t:>+6.2f}, p={p:.3f}) {sig}")

    # Integrity label distribution
    print("\n--- Integrity Label Distribution ---\n")

    single_labels = [r["single"]["integrity_label"] for r in results if r]
    multi_labels = [r["multi"]["integrity_label"] for r in results if r]

    for label in ["fragmented", "living", "rigid"]:
        s_count = single_labels.count(label)
        m_count = multi_labels.count(label)
        print(f"  {label:<12}: Single={s_count}, Multi={m_count}")


def main():
    print("E001 Reanalysis Script")
    print("=" * 50)

    results = []

    for pair in E001_PAIRS:
        print(f"\nPair {pair['pair_id']} ({pair['provocation']})")
        result = analyze_pair(pair["pair_id"], pair["provocation"])
        results.append(result)

    # Print comparison
    print_comparison_table(results)

    # Save results
    output_path = Path(__file__).parent.parent / "experiments" / "analysis" / "E001_model_diversity" / "results_with_trajectory.json"

    output = {
        "experiment_id": "E001",
        "analysis_version": "v0.8.0 (trajectory + integrity)",
        "pairs": results,
        "deltas": compute_deltas(results)
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
