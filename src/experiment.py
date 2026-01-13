"""
Matched-pair experiment runner for MASE.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Runs controlled experiments comparing single-model polyphony
vs multi-model ensemble dialogues on the same provocations.
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any

from .agents import EnsembleConfig
from .orchestrator import DialogueOrchestrator
from .session_logger import SessionLogger
from .metrics import compute_metrics_from_session, MetricsResult


@dataclass
class ConditionResult:
    """Results from one condition of an experiment."""
    condition: str  # "single_model" or "multi_model"
    session_path: str
    metrics: MetricsResult
    total_latency_ms: float
    total_tokens: int


@dataclass
class PairResult:
    """Results from a matched pair experiment."""
    pair_id: str
    provocation_id: str
    provocation_text: str
    seed: int
    single_model: ConditionResult
    multi_model: ConditionResult

    # Deltas (multi - single)
    delta_curvature: float
    delta_alpha: float
    delta_entropy: float


@dataclass
class ExperimentResult:
    """Results from a full experiment."""
    experiment_id: str
    start_time: str
    end_time: str
    n_pairs: int
    pairs: List[PairResult]

    # Aggregate statistics
    mean_delta_curvature: float
    mean_delta_alpha: float
    mean_delta_entropy: float


class ExperimentRunner:
    """
    Runs matched-pair experiments comparing conditions.

    Each pair runs the same provocation with the same seed
    under both single-model and multi-model conditions.
    """

    def __init__(
        self,
        single_model_config_path: Path,
        multi_model_config_path: Path,
        output_dir: Path,
        agents_dir: Optional[Path] = None
    ):
        """
        Initialize experiment runner.

        Args:
            single_model_config_path: Path to single_model.yaml
            multi_model_config_path: Path to multi_model.yaml
            output_dir: Base directory for experiment output
            agents_dir: Optional path to agents directory
        """
        self.single_config = EnsembleConfig.from_yaml(single_model_config_path)
        self.multi_config = EnsembleConfig.from_yaml(multi_model_config_path)
        self.output_dir = Path(output_dir)
        self.agents_dir = agents_dir

        self.single_config_path = str(single_model_config_path)
        self.multi_config_path = str(multi_model_config_path)

    def run_pair(
        self,
        provocation: str,
        seed: int,
        provocation_id: Optional[str] = None,
        max_turns: int = 21
    ) -> PairResult:
        """
        Run a single matched pair experiment.

        Args:
            provocation: The opening provocation text
            seed: Random seed (same for both conditions)
            provocation_id: Optional identifier
            max_turns: Maximum turns per dialogue

        Returns:
            PairResult with metrics from both conditions
        """
        pair_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        pair_dir = self.output_dir / f"pair_{pair_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Matched Pair Experiment: {pair_id}")
        print(f"Provocation: {provocation_id or 'custom'}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")

        # Run single-model condition
        print(f"\n--- Condition 1: Single Model ---")
        single_result = self._run_condition(
            config=self.single_config,
            config_path=self.single_config_path,
            provocation=provocation,
            seed=seed,
            provocation_id=provocation_id,
            max_turns=max_turns,
            output_dir=pair_dir / "single_model"
        )

        # Run multi-model condition
        print(f"\n--- Condition 2: Multi Model ---")
        multi_result = self._run_condition(
            config=self.multi_config,
            config_path=self.multi_config_path,
            provocation=provocation,
            seed=seed,
            provocation_id=provocation_id,
            max_turns=max_turns,
            output_dir=pair_dir / "multi_model"
        )

        # Compute deltas
        delta_kappa = multi_result.metrics.semantic_curvature - single_result.metrics.semantic_curvature
        delta_alpha = multi_result.metrics.dfa_alpha - single_result.metrics.dfa_alpha
        delta_h = multi_result.metrics.entropy_shift - single_result.metrics.entropy_shift

        result = PairResult(
            pair_id=pair_id,
            provocation_id=provocation_id or "custom",
            provocation_text=provocation,
            seed=seed,
            single_model=single_result,
            multi_model=multi_result,
            delta_curvature=delta_kappa,
            delta_alpha=delta_alpha,
            delta_entropy=delta_h
        )

        # Save pair result
        self._save_pair_result(pair_dir, result)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Pair {pair_id} Complete")
        print(f"{'='*60}")
        print(f"\n{'Metric':<20} {'Single':<12} {'Multi':<12} {'Delta':<12}")
        print(f"{'-'*56}")
        print(f"{'Curvature (Δκ)':<20} {single_result.metrics.semantic_curvature:<12.4f} {multi_result.metrics.semantic_curvature:<12.4f} {delta_kappa:+.4f}")
        print(f"{'DFA Alpha (α)':<20} {single_result.metrics.dfa_alpha:<12.4f} {multi_result.metrics.dfa_alpha:<12.4f} {delta_alpha:+.4f}")
        print(f"{'Entropy Shift (ΔH)':<20} {single_result.metrics.entropy_shift:<12.4f} {multi_result.metrics.entropy_shift:<12.4f} {delta_h:+.4f}")
        print(f"\nLatency: Single={single_result.total_latency_ms/1000:.1f}s, Multi={multi_result.total_latency_ms/1000:.1f}s")

        return result

    def _run_condition(
        self,
        config: EnsembleConfig,
        config_path: str,
        provocation: str,
        seed: int,
        provocation_id: Optional[str],
        max_turns: int,
        output_dir: Path
    ) -> ConditionResult:
        """Run one condition and return results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        orchestrator = DialogueOrchestrator(config, self.agents_dir)
        session_path = orchestrator.run_dialogue(
            provocation=provocation,
            output_dir=output_dir,
            max_turns=max_turns,
            seed=seed,
            provocation_id=provocation_id,
            config_path=config_path,
            compute_embeddings=True
        )

        # Load session and compute metrics
        session_data = SessionLogger.load_session(session_path)
        metrics = compute_metrics_from_session(session_data)

        return ConditionResult(
            condition=config.mode,
            session_path=str(session_path),
            metrics=metrics,
            total_latency_ms=session_data.get("total_latency_ms", 0),
            total_tokens=session_data.get("total_tokens", 0)
        )

    def _save_pair_result(self, pair_dir: Path, result: PairResult):
        """Save pair result to JSON."""
        # Convert to serializable format
        data = {
            "pair_id": result.pair_id,
            "provocation_id": result.provocation_id,
            "provocation_text": result.provocation_text,
            "seed": result.seed,
            "single_model": {
                "condition": result.single_model.condition,
                "session_path": result.single_model.session_path,
                "metrics": asdict(result.single_model.metrics),
                "total_latency_ms": result.single_model.total_latency_ms,
                "total_tokens": result.single_model.total_tokens
            },
            "multi_model": {
                "condition": result.multi_model.condition,
                "session_path": result.multi_model.session_path,
                "metrics": asdict(result.multi_model.metrics),
                "total_latency_ms": result.multi_model.total_latency_ms,
                "total_tokens": result.multi_model.total_tokens
            },
            "deltas": {
                "curvature": result.delta_curvature,
                "alpha": result.delta_alpha,
                "entropy": result.delta_entropy
            }
        }

        path = pair_dir / "pair_result.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def run_experiment(
        self,
        provocations: List[Dict[str, str]],
        base_seed: int = 42,
        max_turns: int = 21
    ) -> ExperimentResult:
        """
        Run a full experiment with multiple matched pairs.

        Args:
            provocations: List of dicts with 'id' and 'text' keys
            base_seed: Starting seed (incremented per pair)
            max_turns: Maximum turns per dialogue

        Returns:
            ExperimentResult with all pairs and aggregate statistics
        """
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = self.output_dir / f"experiment_{experiment_id}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        start_time = datetime.now().isoformat()
        pairs = []

        for i, prov in enumerate(provocations):
            seed = base_seed + i
            result = self.run_pair(
                provocation=prov["text"],
                seed=seed,
                provocation_id=prov.get("id"),
                max_turns=max_turns
            )
            pairs.append(result)

        end_time = datetime.now().isoformat()

        # Compute aggregate statistics
        deltas_kappa = [p.delta_curvature for p in pairs]
        deltas_alpha = [p.delta_alpha for p in pairs]
        deltas_h = [p.delta_entropy for p in pairs]

        import numpy as np
        experiment = ExperimentResult(
            experiment_id=experiment_id,
            start_time=start_time,
            end_time=end_time,
            n_pairs=len(pairs),
            pairs=pairs,
            mean_delta_curvature=float(np.mean(deltas_kappa)),
            mean_delta_alpha=float(np.mean(deltas_alpha)),
            mean_delta_entropy=float(np.mean(deltas_h))
        )

        # Save experiment summary
        self._save_experiment_result(experiment_dir, experiment)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Experiment {experiment_id} Complete")
        print(f"{'='*60}")
        print(f"Pairs: {len(pairs)}")
        print(f"\nMean Deltas (Multi - Single):")
        print(f"  Δκ: {experiment.mean_delta_curvature:+.4f}")
        print(f"  α:  {experiment.mean_delta_alpha:+.4f}")
        print(f"  ΔH: {experiment.mean_delta_entropy:+.4f}")

        return experiment

    def _save_experiment_result(self, experiment_dir: Path, result: ExperimentResult):
        """Save experiment summary to JSON."""
        data = {
            "experiment_id": result.experiment_id,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "n_pairs": result.n_pairs,
            "aggregate": {
                "mean_delta_curvature": result.mean_delta_curvature,
                "mean_delta_alpha": result.mean_delta_alpha,
                "mean_delta_entropy": result.mean_delta_entropy
            },
            "pairs": [
                {
                    "pair_id": p.pair_id,
                    "provocation_id": p.provocation_id,
                    "seed": p.seed,
                    "deltas": {
                        "curvature": p.delta_curvature,
                        "alpha": p.delta_alpha,
                        "entropy": p.delta_entropy
                    }
                }
                for p in result.pairs
            ]
        }

        path = experiment_dir / "experiment_summary.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# Convenience function
def run_matched_pair(
    provocation: str,
    seed: int = 42,
    max_turns: int = 21,
    output_dir: Optional[Path] = None
) -> PairResult:
    """
    Convenience function to run a single matched pair.

    Args:
        provocation: Opening provocation text
        seed: Random seed
        max_turns: Maximum turns
        output_dir: Output directory (default: experiments/runs)

    Returns:
        PairResult
    """
    if output_dir is None:
        output_dir = Path("experiments/runs")

    runner = ExperimentRunner(
        single_model_config_path=Path("experiments/config/single_model.yaml"),
        multi_model_config_path=Path("experiments/config/multi_model.yaml"),
        output_dir=output_dir
    )

    return runner.run_pair(provocation, seed, max_turns=max_turns)


# Test if run directly
if __name__ == "__main__":
    import sys

    print("Experiment Runner Test")
    print("=" * 50)

    from .ollama_client import OllamaClient

    if not OllamaClient.is_running():
        print("Error: Ollama not running")
        sys.exit(1)

    # Quick test with 3 turns
    provocation = "What does it mean to truly listen?"

    result = run_matched_pair(
        provocation=provocation,
        seed=42,
        max_turns=3
    )

    print(f"\nTest complete. Pair ID: {result.pair_id}")
