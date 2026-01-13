"""
Resume utilities for MASE experiments.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Helpers for finding and resuming interrupted experiment sessions.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class CheckpointInfo:
    """Information about a checkpoint file."""
    path: Path
    session_id: str
    mode: str
    provocation_id: Optional[str]
    completed_turns: int
    expected_turns: Optional[int]
    is_complete: bool


def find_checkpoints(runs_dir: Path) -> List[CheckpointInfo]:
    """
    Find all checkpoint files in the runs directory.

    Args:
        runs_dir: Path to experiments/runs directory

    Returns:
        List of CheckpointInfo objects
    """
    checkpoints = []

    for checkpoint_path in runs_dir.rglob("*_checkpoint.json"):
        try:
            info = analyze_checkpoint(checkpoint_path)
            if info:
                checkpoints.append(info)
        except Exception:
            continue

    # Sort by most recent first
    checkpoints.sort(key=lambda x: x.path.stat().st_mtime, reverse=True)
    return checkpoints


def analyze_checkpoint(path: Path) -> Optional[CheckpointInfo]:
    """
    Analyze a checkpoint file to determine its state.

    Args:
        path: Path to checkpoint JSON file

    Returns:
        CheckpointInfo or None if invalid
    """
    try:
        with open(path) as f:
            data = json.load(f)

        completed_turns = len(data.get("turns", []))

        # Check for corresponding completed session
        session_path = path.parent / path.name.replace("_checkpoint", "")
        is_complete = session_path.exists()

        return CheckpointInfo(
            path=path,
            session_id=data.get("session_id", "unknown"),
            mode=data.get("mode", "unknown"),
            provocation_id=data.get("provocation_id"),
            completed_turns=completed_turns,
            expected_turns=None,  # Can't know from checkpoint alone
            is_complete=is_complete
        )
    except Exception:
        return None


def find_incomplete_pairs(runs_dir: Path) -> List[Dict[str, Any]]:
    """
    Find pair directories where one or both conditions are incomplete.

    Args:
        runs_dir: Path to experiments/runs directory

    Returns:
        List of dicts with pair info and missing conditions
    """
    incomplete = []

    for pair_dir in runs_dir.glob("pair_*"):
        if not pair_dir.is_dir():
            continue

        single_dir = pair_dir / "single_model"
        multi_dir = pair_dir / "multi_model"

        single_complete = _has_complete_session(single_dir)
        multi_complete = _has_complete_session(multi_dir)

        if not single_complete or not multi_complete:
            # Load pair info if available
            pair_result = pair_dir / "pair_result.json"
            pair_info = {}
            if pair_result.exists():
                try:
                    with open(pair_result) as f:
                        pair_info = json.load(f)
                except:
                    pass

            incomplete.append({
                "pair_dir": pair_dir,
                "pair_id": pair_dir.name.replace("pair_", ""),
                "provocation_id": pair_info.get("provocation_id", "unknown"),
                "seed": pair_info.get("seed"),
                "single_model_complete": single_complete,
                "multi_model_complete": multi_complete,
                "single_checkpoint": _find_latest_checkpoint(single_dir),
                "multi_checkpoint": _find_latest_checkpoint(multi_dir)
            })

    return incomplete


def _has_complete_session(condition_dir: Path) -> bool:
    """Check if a condition directory has a completed session."""
    if not condition_dir.exists():
        return False

    # Look for session JSON without _checkpoint suffix
    for f in condition_dir.glob("session_*.json"):
        if "_checkpoint" not in f.name:
            return True
    return False


def _find_latest_checkpoint(condition_dir: Path) -> Optional[Path]:
    """Find the most recent checkpoint in a condition directory."""
    if not condition_dir.exists():
        return None

    checkpoints = list(condition_dir.glob("session_*_checkpoint.json"))
    if not checkpoints:
        return None

    # Return most recently modified
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def print_status(runs_dir: Path):
    """Print status of all experiments in runs directory."""
    print("\n" + "=" * 60)
    print("MASE Experiment Status")
    print("=" * 60)

    incomplete = find_incomplete_pairs(runs_dir)

    if not incomplete:
        print("\nAll pairs complete.")
    else:
        print(f"\n{len(incomplete)} incomplete pairs found:\n")

        for p in incomplete:
            print(f"Pair: {p['pair_id']}")
            print(f"  Provocation: {p['provocation_id']}")
            print(f"  Single model: {'Complete' if p['single_model_complete'] else 'INCOMPLETE'}")
            print(f"  Multi model:  {'Complete' if p['multi_model_complete'] else 'INCOMPLETE'}")

            if p['single_checkpoint']:
                info = analyze_checkpoint(p['single_checkpoint'])
                if info:
                    print(f"    Single checkpoint: {info.completed_turns} turns")

            if p['multi_checkpoint']:
                info = analyze_checkpoint(p['multi_checkpoint'])
                if info:
                    print(f"    Multi checkpoint: {info.completed_turns} turns")

            print()

    print("=" * 60)


# CLI entry point
if __name__ == "__main__":
    import sys

    runs_dir = Path("experiments/runs")
    if len(sys.argv) > 1:
        runs_dir = Path(sys.argv[1])

    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        sys.exit(1)

    print_status(runs_dir)
