"""
Dialectical Test Runner for MASE.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Automated test-eval-refine loop for improving Socratic dialectical quality.
Runs sessions with variant configurations, collects dialectical metrics,
and compares against baseline.

Usage:
    from src.dialectical_test import DialecticalTestRunner

    runner = DialecticalTestRunner()
    result = runner.run_comparison(
        variant_name="v001_negative_examples",
        provocation_id="p001_success",
        n_runs=3
    )
"""

import json
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from copy import deepcopy

# Handle both package and direct execution
try:
    from .agents import EnsembleConfig, load_ensemble, Agent
    from .orchestrator import DialogueOrchestrator
    from .session_logger import SessionLogger
    from .session_analysis import (
        analyze_session,
        SessionAnalysisResult,
        compute_challenge_density,
        compute_politeness_overhead,
        compute_refuting_question_rate,
        compute_turn_type_distribution,
        compute_antithesis_ratio
    )
except ImportError:
    from agents import EnsembleConfig, load_ensemble, Agent
    from orchestrator import DialogueOrchestrator
    from session_logger import SessionLogger
    from session_analysis import (
        analyze_session,
        SessionAnalysisResult,
        compute_challenge_density,
        compute_politeness_overhead,
        compute_refuting_question_rate,
        compute_turn_type_distribution,
        compute_antithesis_ratio
    )


@dataclass
class DialecticalMetrics:
    """Dialectical quality metrics for a single session."""
    session_id: str
    session_path: str

    # Core dialectical metrics
    antithesis_ratio: float
    challenge_density: float
    politeness_overhead: float
    refuting_question_rate: float

    # Turn type distribution
    thesis_count: int
    antithesis_count: int
    synthesis_count: int
    build_count: int
    question_count: int

    # Derived
    build_turns_pct: float
    dialectical_score: float  # Composite score

    # Context
    n_turns: int
    provocation_id: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VariantRunResult:
    """Results from running a variant N times."""
    variant_name: str
    n_runs: int
    provocation_id: str
    provocation_text: str

    # Individual run metrics
    runs: List[DialecticalMetrics]

    # Aggregated metrics (mean ± std)
    mean_antithesis_ratio: float
    std_antithesis_ratio: float
    mean_challenge_density: float
    std_challenge_density: float
    mean_politeness_overhead: float
    std_politeness_overhead: float
    mean_build_turns_pct: float
    std_build_turns_pct: float
    mean_dialectical_score: float
    std_dialectical_score: float

    def to_dict(self) -> dict:
        result = asdict(self)
        result['runs'] = [r.to_dict() for r in self.runs]
        return result


@dataclass
class ComparisonResult:
    """Results from comparing a variant against baseline."""
    test_id: str
    timestamp: str
    variant_name: str
    baseline_name: str
    provocation_id: str
    n_runs: int

    # Results per condition
    baseline_result: VariantRunResult
    variant_result: VariantRunResult

    # Deltas (variant - baseline)
    delta_antithesis_ratio: float
    delta_challenge_density: float
    delta_politeness_overhead: float
    delta_build_turns_pct: float
    delta_dialectical_score: float

    # Effect assessment
    improved_metrics: List[str]
    degraded_metrics: List[str]
    verdict: str  # "improved", "degraded", "mixed", "no_change"

    def to_dict(self) -> dict:
        result = {
            'test_id': self.test_id,
            'timestamp': self.timestamp,
            'variant_name': self.variant_name,
            'baseline_name': self.baseline_name,
            'provocation_id': self.provocation_id,
            'n_runs': self.n_runs,
            'baseline_result': self.baseline_result.to_dict(),
            'variant_result': self.variant_result.to_dict(),
            'deltas': {
                'antithesis_ratio': self.delta_antithesis_ratio,
                'challenge_density': self.delta_challenge_density,
                'politeness_overhead': self.delta_politeness_overhead,
                'build_turns_pct': self.delta_build_turns_pct,
                'dialectical_score': self.delta_dialectical_score
            },
            'improved_metrics': self.improved_metrics,
            'degraded_metrics': self.degraded_metrics,
            'verdict': self.verdict
        }
        return result


class VariantLoader:
    """Loads and applies variant configurations."""

    def __init__(self, variants_dir: Path):
        self.variants_dir = Path(variants_dir)

    def load_variant(self, name: str) -> dict:
        """Load a variant definition from YAML."""
        path = self.variants_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Variant not found: {path}")

        with open(path) as f:
            return yaml.safe_load(f)

    def get_modifications(self, variant: dict) -> List[dict]:
        """Extract modifications from variant definition."""
        return variant.get('modifications', [])

    def apply_prompt_modification(
        self,
        base_prompt: str,
        modification: dict
    ) -> str:
        """Apply a prompt modification."""
        mod_type = modification.get('type')
        content = modification.get('content', '')
        position = modification.get('position', 'after')
        target = modification.get('target', '')

        if mod_type == 'prompt_addition':
            if target == 'dialectical_norms':
                # Find the DIALECTICAL NORMS section and add after it
                marker = "DIALECTICAL NORMS:"
                if marker in base_prompt:
                    # Find the end of the dialectical norms section
                    idx = base_prompt.find(marker)
                    # Find the next section (starts with caps and colon)
                    remaining = base_prompt[idx + len(marker):]

                    # Look for next section header
                    import re
                    next_section = re.search(r'\n[A-Z][A-Z\s]+:', remaining)
                    if next_section:
                        insert_point = idx + len(marker) + next_section.start()
                        return base_prompt[:insert_point] + content + base_prompt[insert_point:]
                    else:
                        # Add at end
                        return base_prompt + content

            elif position == 'after':
                return base_prompt + content
            elif position == 'before':
                return content + base_prompt

        elif mod_type == 'prompt_replace':
            old = modification.get('old', '')
            new = modification.get('new', '')
            return base_prompt.replace(old, new)

        return base_prompt

    def apply_agent_modification(
        self,
        agent: Agent,
        modification: dict
    ) -> Agent:
        """Apply a modification to an agent's prompt."""
        target_agent = modification.get('target', '')
        if agent.id != target_agent:
            return agent

        content = modification.get('content', '')
        position = modification.get('position', 'end')

        # Create modified copy
        modified = deepcopy(agent)
        if position == 'end':
            modified.system_prompt = agent.system_prompt + content
        elif position == 'start':
            modified.system_prompt = content + agent.system_prompt

        return modified


class DialecticalTestRunner:
    """
    Runs dialectical quality tests comparing variants against baseline.

    Test cycle:
    1. Load variant definition
    2. Run N sessions with baseline config
    3. Run N sessions with variant modifications
    4. Collect dialectical metrics
    5. Compare and generate report
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        variants_dir: Optional[Path] = None,
        provocations_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        agents_dir: Optional[Path] = None
    ):
        """
        Initialize the test runner.

        Args:
            config_path: Path to ensemble config YAML
            variants_dir: Path to variant definitions
            provocations_path: Path to provocations YAML
            output_dir: Where to save test results
            agents_dir: Path to agent persona definitions
        """
        base_path = Path(__file__).parent.parent

        self.config_path = config_path or base_path / "experiments/config/multi_model.yaml"
        self.variants_dir = variants_dir or base_path / "experiments/dialectical/variants"
        self.provocations_path = provocations_path or base_path / "experiments/provocations/seed_provocations.yaml"
        self.output_dir = output_dir or base_path / "experiments/dialectical/runs"
        self.agents_dir = agents_dir or base_path / "agents/personas"

        self.variant_loader = VariantLoader(self.variants_dir)
        self._load_provocations()

    def _load_provocations(self):
        """Load seed provocations."""
        with open(self.provocations_path) as f:
            data = yaml.safe_load(f)
        self.provocations = {
            p['id']: p['text'].strip()
            for p in data.get('provocations', [])
        }

    def _compute_dialectical_metrics(
        self,
        session_path: Path,
        provocation_id: str
    ) -> DialecticalMetrics:
        """Compute dialectical metrics for a session."""
        # Use existing session analysis
        result = analyze_session(session_path, compute_embeddings=True)

        # Get turn type distribution
        turn_dist = result.turn_type_distribution or {}
        total_turns = sum(turn_dist.values()) if turn_dist else 1

        build_count = turn_dist.get('BUILD', 0)
        build_pct = build_count / total_turns if total_turns > 0 else 0

        # Compute composite dialectical score
        # Higher is better: more challenge, less politeness, more antithesis
        dialectical_score = (
            (result.antithesis_ratio or 0) * 2 +
            (result.challenge_density or 0) +
            (result.refuting_question_rate or 0) -
            (result.politeness_overhead or 0) -
            build_pct
        )

        return DialecticalMetrics(
            session_id=session_path.stem,
            session_path=str(session_path),
            antithesis_ratio=result.antithesis_ratio or 0,
            challenge_density=result.challenge_density or 0,
            politeness_overhead=result.politeness_overhead or 0,
            refuting_question_rate=result.refuting_question_rate or 0,
            thesis_count=turn_dist.get('THESIS', 0),
            antithesis_count=turn_dist.get('ANTITHESIS', 0),
            synthesis_count=turn_dist.get('SYNTHESIS', 0),
            build_count=build_count,
            question_count=turn_dist.get('QUESTION', 0),
            build_turns_pct=build_pct,
            dialectical_score=dialectical_score,
            n_turns=result.n_turns,
            provocation_id=provocation_id
        )

    def _aggregate_metrics(
        self,
        runs: List[DialecticalMetrics],
        variant_name: str,
        provocation_id: str,
        provocation_text: str
    ) -> VariantRunResult:
        """Aggregate metrics across multiple runs."""
        antithesis_ratios = [r.antithesis_ratio for r in runs]
        challenge_densities = [r.challenge_density for r in runs]
        politeness_overheads = [r.politeness_overhead for r in runs]
        build_pcts = [r.build_turns_pct for r in runs]
        dialectical_scores = [r.dialectical_score for r in runs]

        return VariantRunResult(
            variant_name=variant_name,
            n_runs=len(runs),
            provocation_id=provocation_id,
            provocation_text=provocation_text,
            runs=runs,
            mean_antithesis_ratio=float(np.mean(antithesis_ratios)),
            std_antithesis_ratio=float(np.std(antithesis_ratios)),
            mean_challenge_density=float(np.mean(challenge_densities)),
            std_challenge_density=float(np.std(challenge_densities)),
            mean_politeness_overhead=float(np.mean(politeness_overheads)),
            std_politeness_overhead=float(np.std(politeness_overheads)),
            mean_build_turns_pct=float(np.mean(build_pcts)),
            std_build_turns_pct=float(np.std(build_pcts)),
            mean_dialectical_score=float(np.mean(dialectical_scores)),
            std_dialectical_score=float(np.std(dialectical_scores))
        )

    def _run_sessions(
        self,
        variant_name: str,
        variant_mods: List[dict],
        provocation_id: str,
        n_runs: int,
        max_turns: int,
        base_seed: int,
        run_dir: Path
    ) -> List[DialecticalMetrics]:
        """Run N sessions with a variant configuration."""
        results = []
        provocation_text = self.provocations[provocation_id]

        # Extract prompt-level additions from modifications
        prompt_additions = ""
        for mod in variant_mods:
            if mod.get('type') == 'prompt_addition':
                prompt_additions += mod.get('content', '')

        for i in range(n_runs):
            seed = base_seed + i
            session_dir = run_dir / f"run_{i:02d}"
            session_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Run {i+1}/{n_runs} (seed={seed})...")

            # Load config and agents
            config = EnsembleConfig.from_yaml(self.config_path)
            agents = load_ensemble(self.agents_dir, config)

            # Apply agent-level modifications
            for mod in variant_mods:
                if mod.get('type') == 'agent_prompt_addition':
                    target = mod.get('target')
                    if target in agents:
                        agents[target] = self.variant_loader.apply_agent_modification(
                            agents[target], mod
                        )

            # Create orchestrator with modified agents and prompt additions
            orchestrator = DialogueOrchestrator(
                config,
                self.agents_dir,
                turn_retries=3,
                turn_retry_backoff=2.0,
                keep_models_warm=True,
                prompt_additions=prompt_additions if prompt_additions else None
            )
            # Replace agents with modified versions
            orchestrator.agents = agents

            # Run dialogue
            session_path = orchestrator.run_dialogue(
                provocation=provocation_text,
                output_dir=session_dir,
                max_turns=max_turns,
                seed=seed,
                provocation_id=provocation_id,
                config_path=str(self.config_path),
                compute_embeddings=True
            )

            # Compute metrics
            metrics = self._compute_dialectical_metrics(session_path, provocation_id)
            results.append(metrics)

            print(f"    antithesis_ratio={metrics.antithesis_ratio:.2f}, "
                  f"challenge_density={metrics.challenge_density:.2f}, "
                  f"dialectical_score={metrics.dialectical_score:.2f}")

        return results

    def run_comparison(
        self,
        variant_name: str,
        provocation_id: str,
        n_runs: int = 3,
        max_turns: int = 15,
        base_seed: int = 42
    ) -> ComparisonResult:
        """
        Run a comparison between baseline and a variant.

        Args:
            variant_name: Name of variant to test (without .yaml)
            provocation_id: Provocation to use (e.g., "p001_success")
            n_runs: Number of sessions per condition
            max_turns: Maximum turns per session
            base_seed: Base random seed

        Returns:
            ComparisonResult with metrics and verdict
        """
        if provocation_id not in self.provocations:
            raise ValueError(f"Unknown provocation: {provocation_id}")

        # Load variant
        variant = self.variant_loader.load_variant(variant_name)
        variant_mods = self.variant_loader.get_modifications(variant)
        baseline_name = variant.get('baseline', 'baseline')

        # Create test directory
        test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = self.output_dir / f"dtest_{test_id}_{variant_name}"
        test_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Dialectical Test: {variant_name} vs {baseline_name}")
        print(f"Provocation: {provocation_id}")
        print(f"Runs per condition: {n_runs}")
        print(f"{'='*60}")

        # Run baseline
        print(f"\n--- Baseline ({baseline_name}) ---")
        baseline_runs = self._run_sessions(
            variant_name=baseline_name,
            variant_mods=[],  # No modifications for baseline
            provocation_id=provocation_id,
            n_runs=n_runs,
            max_turns=max_turns,
            base_seed=base_seed,
            run_dir=test_dir / "baseline"
        )
        baseline_result = self._aggregate_metrics(
            baseline_runs, baseline_name, provocation_id,
            self.provocations[provocation_id]
        )

        # Run variant
        print(f"\n--- Variant ({variant_name}) ---")
        variant_runs = self._run_sessions(
            variant_name=variant_name,
            variant_mods=variant_mods,
            provocation_id=provocation_id,
            n_runs=n_runs,
            max_turns=max_turns,
            base_seed=base_seed + 100,  # Different seed range for variant
            run_dir=test_dir / "variant"
        )
        variant_result = self._aggregate_metrics(
            variant_runs, variant_name, provocation_id,
            self.provocations[provocation_id]
        )

        # Compute deltas
        delta_antithesis = variant_result.mean_antithesis_ratio - baseline_result.mean_antithesis_ratio
        delta_challenge = variant_result.mean_challenge_density - baseline_result.mean_challenge_density
        delta_politeness = variant_result.mean_politeness_overhead - baseline_result.mean_politeness_overhead
        delta_build = variant_result.mean_build_turns_pct - baseline_result.mean_build_turns_pct
        delta_dialectical = variant_result.mean_dialectical_score - baseline_result.mean_dialectical_score

        # Assess improvement
        # For dialectical quality: higher antithesis/challenge is better, lower politeness/build is better
        improved = []
        degraded = []

        if delta_antithesis > 0.02:
            improved.append('antithesis_ratio')
        elif delta_antithesis < -0.02:
            degraded.append('antithesis_ratio')

        if delta_challenge > 0.02:
            improved.append('challenge_density')
        elif delta_challenge < -0.02:
            degraded.append('challenge_density')

        if delta_politeness < -0.02:
            improved.append('politeness_overhead')
        elif delta_politeness > 0.02:
            degraded.append('politeness_overhead')

        if delta_build < -0.02:
            improved.append('build_turns_pct')
        elif delta_build > 0.02:
            degraded.append('build_turns_pct')

        # Determine verdict
        if len(improved) > len(degraded) and len(degraded) == 0:
            verdict = "improved"
        elif len(degraded) > len(improved) and len(improved) == 0:
            verdict = "degraded"
        elif len(improved) > 0 or len(degraded) > 0:
            verdict = "mixed"
        else:
            verdict = "no_change"

        result = ComparisonResult(
            test_id=test_id,
            timestamp=datetime.now().isoformat(),
            variant_name=variant_name,
            baseline_name=baseline_name,
            provocation_id=provocation_id,
            n_runs=n_runs,
            baseline_result=baseline_result,
            variant_result=variant_result,
            delta_antithesis_ratio=delta_antithesis,
            delta_challenge_density=delta_challenge,
            delta_politeness_overhead=delta_politeness,
            delta_build_turns_pct=delta_build,
            delta_dialectical_score=delta_dialectical,
            improved_metrics=improved,
            degraded_metrics=degraded,
            verdict=verdict
        )

        # Save results
        self._save_result(test_dir, result)

        # Print summary
        self._print_summary(result)

        return result

    def _save_result(self, test_dir: Path, result: ComparisonResult):
        """Save comparison result to JSON."""
        path = test_dir / "comparison.json"
        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    def _print_summary(self, result: ComparisonResult):
        """Print a summary of the comparison."""
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {result.variant_name} vs {result.baseline_name}")
        print(f"{'='*60}")

        print(f"\n{'Metric':<25} {'Baseline':<12} {'Variant':<12} {'Delta':<12}")
        print(f"{'-'*61}")

        b = result.baseline_result
        v = result.variant_result

        print(f"{'Antithesis Ratio':<25} {b.mean_antithesis_ratio:.3f}±{b.std_antithesis_ratio:.3f}   "
              f"{v.mean_antithesis_ratio:.3f}±{v.std_antithesis_ratio:.3f}   "
              f"{result.delta_antithesis_ratio:+.3f}")

        print(f"{'Challenge Density':<25} {b.mean_challenge_density:.3f}±{b.std_challenge_density:.3f}   "
              f"{v.mean_challenge_density:.3f}±{v.std_challenge_density:.3f}   "
              f"{result.delta_challenge_density:+.3f}")

        print(f"{'Politeness Overhead':<25} {b.mean_politeness_overhead:.3f}±{b.std_politeness_overhead:.3f}   "
              f"{v.mean_politeness_overhead:.3f}±{v.std_politeness_overhead:.3f}   "
              f"{result.delta_politeness_overhead:+.3f}")

        print(f"{'BUILD Turns %':<25} {b.mean_build_turns_pct:.3f}±{b.std_build_turns_pct:.3f}   "
              f"{v.mean_build_turns_pct:.3f}±{v.std_build_turns_pct:.3f}   "
              f"{result.delta_build_turns_pct:+.3f}")

        print(f"{'Dialectical Score':<25} {b.mean_dialectical_score:.3f}±{b.std_dialectical_score:.3f}   "
              f"{v.mean_dialectical_score:.3f}±{v.std_dialectical_score:.3f}   "
              f"{result.delta_dialectical_score:+.3f}")

        print(f"\n{'='*60}")
        print(f"VERDICT: {result.verdict.upper()}")
        if result.improved_metrics:
            print(f"  Improved: {', '.join(result.improved_metrics)}")
        if result.degraded_metrics:
            print(f"  Degraded: {', '.join(result.degraded_metrics)}")
        print(f"{'='*60}")


def run_quick_test(variant_name: str, provocation_id: str = "p001_success"):
    """
    Quick test with 1 run per condition for fast iteration.

    Args:
        variant_name: Variant to test
        provocation_id: Provocation to use
    """
    runner = DialecticalTestRunner()
    return runner.run_comparison(
        variant_name=variant_name,
        provocation_id=provocation_id,
        n_runs=1,
        max_turns=10
    )


def run_full_test(variant_name: str, provocation_id: str = "p001_success"):
    """
    Full test with 3 runs per condition for statistical robustness.

    Args:
        variant_name: Variant to test
        provocation_id: Provocation to use
    """
    runner = DialecticalTestRunner()
    return runner.run_comparison(
        variant_name=variant_name,
        provocation_id=provocation_id,
        n_runs=3,
        max_turns=15
    )


# CLI interface
if __name__ == "__main__":
    import sys

    print("Dialectical Test Runner")
    print("=" * 50)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python -m src.dialectical_test <variant_name> [provocation_id] [n_runs]")
        print("\nExample:")
        print("  python -m src.dialectical_test v001_negative_examples p001_success 3")
        print("\nAvailable variants:")
        variants_dir = Path(__file__).parent.parent / "experiments/dialectical/variants"
        for v in sorted(variants_dir.glob("*.yaml")):
            print(f"  - {v.stem}")
        sys.exit(0)

    variant_name = sys.argv[1]
    provocation_id = sys.argv[2] if len(sys.argv) > 2 else "p001_success"
    n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    runner = DialecticalTestRunner()
    result = runner.run_comparison(
        variant_name=variant_name,
        provocation_id=provocation_id,
        n_runs=n_runs
    )

    print(f"\nResults saved to: experiments/dialectical/runs/dtest_{result.test_id}_{variant_name}/")
