#!/usr/bin/env python3
"""
Resume failed MASE experiments.

Finds incomplete experiment pairs and resumes them from their checkpoints.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent to path so we can import src as a package
sys.path.insert(0, str(Path(__file__).parent))

from src.ollama_client import OllamaClient
from src.agents import EnsembleConfig
from src.orchestrator import DialogueOrchestrator
from src.session_logger import SessionLogger
from src.metrics import compute_metrics_from_session
from src.resume import find_incomplete_pairs, analyze_checkpoint


def load_provocation(provocation_id: str) -> str:
    """Load provocation text from seed_provocations.yaml."""
    import yaml

    prov_path = Path("experiments/provocations/seed_provocations.yaml")
    with open(prov_path) as f:
        data = yaml.safe_load(f)

    for prov in data.get("provocations", []):
        if prov["id"] == provocation_id:
            return prov["text"].strip()

    raise ValueError(f"Provocation not found: {provocation_id}")


def get_session_info(session_dir: Path) -> dict:
    """Get seed and provocation info from a completed session."""
    for f in session_dir.glob("session_*.json"):
        if "_checkpoint" not in f.name:
            with open(f) as fp:
                data = json.load(fp)
                return {
                    "seed": data.get("seed"),
                    "provocation_id": data.get("provocation_id"),
                    "provocation_text": data.get("provocation_text"),
                    "turns": len(data.get("turns", []))
                }
    return {}


def resume_pair(pair_info: dict, max_turns: int = 21) -> bool:
    """
    Resume an incomplete pair from checkpoint.

    Returns True if successful, False otherwise.
    """
    pair_dir = Path(pair_info["pair_dir"])
    checkpoint_path = pair_info.get("multi_checkpoint")

    if not checkpoint_path:
        print(f"  No checkpoint found for {pair_dir.name}")
        return False

    checkpoint_path = Path(checkpoint_path)

    # Get info from the single-model session (which completed)
    single_info = get_session_info(pair_dir / "single_model")
    if not single_info:
        print(f"  Could not load single-model session info")
        return False

    seed = single_info["seed"]
    provocation_id = single_info["provocation_id"]
    provocation_text = single_info["provocation_text"]

    print(f"\n{'='*60}")
    print(f"Resuming: {pair_dir.name}")
    print(f"Provocation: {provocation_id}")
    print(f"Seed: {seed}")

    # Load checkpoint to see progress
    checkpoint_info = analyze_checkpoint(checkpoint_path)
    if checkpoint_info:
        print(f"Checkpoint: {checkpoint_info.completed_turns} turns completed")

    print(f"{'='*60}\n")

    # Load multi-model config
    config = EnsembleConfig.from_yaml(Path("experiments/config/multi_model.yaml"))

    # Create orchestrator
    orchestrator = DialogueOrchestrator(
        config,
        agents_dir=Path(".claude/agents"),
        turn_retries=3,
        turn_retry_backoff=2.0,
        keep_models_warm=True
    )

    # Resume from checkpoint
    output_dir = pair_dir / "multi_model"
    session_path = orchestrator.run_dialogue(
        provocation=provocation_text,
        output_dir=output_dir,
        max_turns=max_turns,
        seed=seed,
        provocation_id=provocation_id,
        config_path="experiments/config/multi_model.yaml",
        compute_embeddings=True,
        resume_from=checkpoint_path
    )

    print(f"\nSession saved to: {session_path}")

    # Compute metrics and update pair result
    try:
        update_pair_result(pair_dir, single_info, session_path, provocation_id, provocation_text, seed)
    except Exception as e:
        print(f"Warning: Could not update pair result: {e}")

    return True


def update_pair_result(
    pair_dir: Path,
    single_info: dict,
    multi_session_path: Path,
    provocation_id: str,
    provocation_text: str,
    seed: int
):
    """Update the pair_result.json with completed results."""
    # Load single-model metrics
    single_session_path = None
    for f in (pair_dir / "single_model").glob("session_*.json"):
        if "_checkpoint" not in f.name:
            single_session_path = f
            break

    if not single_session_path:
        print("  Could not find single-model session for metrics")
        return

    single_data = SessionLogger.load_session(single_session_path)
    multi_data = SessionLogger.load_session(multi_session_path)

    single_metrics = compute_metrics_from_session(single_data)
    multi_metrics = compute_metrics_from_session(multi_data)

    # Compute deltas
    delta_kappa = multi_metrics.semantic_curvature - single_metrics.semantic_curvature
    delta_alpha = multi_metrics.dfa_alpha - single_metrics.dfa_alpha
    delta_h = multi_metrics.entropy_shift - single_metrics.entropy_shift

    # Build result
    from dataclasses import asdict
    result = {
        "pair_id": pair_dir.name.replace("pair_", ""),
        "provocation_id": provocation_id,
        "provocation_text": provocation_text,
        "seed": seed,
        "single_model": {
            "condition": "single_model",
            "session_path": str(single_session_path),
            "metrics": asdict(single_metrics),
            "total_latency_ms": single_data.get("total_latency_ms", 0),
            "total_tokens": single_data.get("total_tokens", 0)
        },
        "multi_model": {
            "condition": "multi_model",
            "session_path": str(multi_session_path),
            "metrics": asdict(multi_metrics),
            "total_latency_ms": multi_data.get("total_latency_ms", 0),
            "total_tokens": multi_data.get("total_tokens", 0)
        },
        "deltas": {
            "curvature": delta_kappa,
            "alpha": delta_alpha,
            "entropy": delta_h
        }
    }

    # Save
    result_path = pair_dir / "pair_result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Pair Result Updated")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Single':<12} {'Multi':<12} {'Delta':<12}")
    print(f"{'-'*56}")
    print(f"{'Curvature (Δκ)':<20} {single_metrics.semantic_curvature:<12.4f} {multi_metrics.semantic_curvature:<12.4f} {delta_kappa:+.4f}")
    print(f"{'DFA Alpha (α)':<20} {single_metrics.dfa_alpha:<12.4f} {multi_metrics.dfa_alpha:<12.4f} {delta_alpha:+.4f}")
    print(f"{'Entropy Shift (ΔH)':<20} {single_metrics.entropy_shift:<12.4f} {multi_metrics.entropy_shift:<12.4f} {delta_h:+.4f}")
    print(f"\nSaved to: {result_path}")


def main():
    # Check Ollama
    if not OllamaClient.is_running():
        print("Error: Ollama not running. Start with: ollama serve")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("MASE Experiment Resume")
    print("=" * 60)

    runs_dir = Path("experiments/runs")
    incomplete = find_incomplete_pairs(runs_dir)

    if not incomplete:
        print("\nNo incomplete experiments found.")
        return

    print(f"\nFound {len(incomplete)} incomplete pair(s):")
    for p in incomplete:
        checkpoint_info = None
        if p.get("multi_checkpoint"):
            checkpoint_info = analyze_checkpoint(Path(p["multi_checkpoint"]))

        turns_info = f"{checkpoint_info.completed_turns}/21 turns" if checkpoint_info else "unknown"
        print(f"  - {p['pair_id']}: {turns_info}")

    # Ask for confirmation
    print(f"\nResume all incomplete experiments? [y/N] ", end="")
    response = input().strip().lower()

    if response != 'y':
        print("Aborted.")
        return

    # Resume each incomplete pair
    # Sort by pair_id to get the most recent attempts
    incomplete.sort(key=lambda x: x['pair_id'], reverse=True)

    # Track unique provocation/seed combos to avoid duplicates
    seen = set()
    resumed = 0

    for p in incomplete:
        # Get session info to check for duplicates
        single_info = get_session_info(Path(p["pair_dir"]) / "single_model")
        key = (single_info.get("provocation_id"), single_info.get("seed"))

        if key in seen:
            print(f"\nSkipping {p['pair_id']} (duplicate of already resumed experiment)")
            continue

        seen.add(key)

        try:
            if resume_pair(p):
                resumed += 1
        except Exception as e:
            print(f"\nError resuming {p['pair_id']}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Resume Complete: {resumed} experiment(s) resumed")
    print("=" * 60)


if __name__ == "__main__":
    main()
