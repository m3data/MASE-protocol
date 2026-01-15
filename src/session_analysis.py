"""
Session Analysis Module for MASE.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Provides post-hoc and streaming analysis of MASE dialogue sessions,
integrating metrics computation with basin detection and trajectory integrity.

Usage:
    from src.session_analysis import analyze_session, SessionAnalyzer

    # Quick analysis of a session file
    results = analyze_session(Path("experiments/runs/session.json"))

    # Streaming analysis during dialogue
    analyzer = SessionAnalyzer()
    for turn in dialogue:
        state = analyzer.process_turn(turn)
        print(f"Basin: {state['basin']}, Integrity: {state.trajectory_integrity}")
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

# Handle both package and direct execution
try:
    from .metrics import (
        compute_metrics,
        compute_metrics_with_ci,
        semantic_curvature,
        dfa_alpha,
        entropy_shift,
        semantic_velocity,
        MetricsResultWithCI,
        THRESHOLDS
    )
    from .basins import (
        BasinDetector,
        BasinHistory,
        DialogueContext,
        compute_psi_vector,
        compute_dialogue_context
    )
    from .affective import (
        compute_affective_substrate,
        compute_agent_affective_divergence,
        AffectiveResult
    )
    from .trajectory import (
        TrajectoryBuffer,
        TrajectoryStateVector,
        compute_trajectory_derivatives
    )
    from .integrity import (
        IntegrityAnalyzer,
        TransformationDetector,
        IntegrityResult
    )
except ImportError:
    from metrics import (
        compute_metrics,
        compute_metrics_with_ci,
        semantic_curvature,
        dfa_alpha,
        entropy_shift,
        semantic_velocity,
        MetricsResultWithCI,
        THRESHOLDS
    )
    from basins import (
        BasinDetector,
        BasinHistory,
        DialogueContext,
        compute_psi_vector,
        compute_dialogue_context
    )
    from affective import (
        compute_affective_substrate,
        compute_agent_affective_divergence,
        AffectiveResult
    )
    from trajectory import (
        TrajectoryBuffer,
        TrajectoryStateVector,
        compute_trajectory_derivatives
    )
    from integrity import (
        IntegrityAnalyzer,
        TransformationDetector,
        IntegrityResult
    )


@dataclass
class TurnState:
    """State snapshot for a single turn."""
    turn_number: int
    agent_id: str
    basin: str
    basin_confidence: float
    psi_semantic: float
    psi_temporal: float
    psi_affective: float
    coherence_pattern: str
    residence_time: int
    # Trajectory dynamics (from TrajectoryBuffer)
    velocity_magnitude: Optional[float] = None
    acceleration_magnitude: Optional[float] = None
    trajectory_curvature: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionAnalysisResult:
    """Complete analysis results for a session."""
    # Core metrics (whole session)
    semantic_curvature: float
    dfa_alpha: float
    entropy_shift: float
    semantic_velocity_mean: float

    # Basin analysis
    basin_sequence: List[str]
    basin_distribution: Dict[str, int]
    transition_count: int
    dominant_basin: str
    dominant_basin_percentage: float

    # Quality indicators
    voice_distinctiveness: float
    coherence_pattern_distribution: Dict[str, int]
    inquiry_vs_mimicry_ratio: float

    # Per-turn states (for visualization)
    turn_states: List[TurnState]

    # Metadata
    n_turns: int
    agents: List[str]

    # Confidence intervals (optional, populated when compute_ci=True)
    curvature_ci: Optional[Tuple[float, float]] = None
    alpha_ci: Optional[Tuple[float, float]] = None
    entropy_ci: Optional[Tuple[float, float]] = None

    # Significance indicators (optional)
    curvature_p_value: Optional[float] = None
    alpha_r_squared: Optional[float] = None
    entropy_stability: Optional[float] = None

    # Threshold flags (Morgoulis 2025)
    curvature_significant: Optional[bool] = None
    alpha_in_range: Optional[bool] = None
    entropy_significant: Optional[bool] = None

    # Affective analysis (optional, from affective.py)
    psi_affective: Optional[float] = None
    sentiment_mean: Optional[float] = None
    sentiment_variance: Optional[float] = None
    hedging_density: Optional[float] = None
    agent_sentiment: Optional[Dict[str, float]] = None
    agent_affective_divergence: Optional[float] = None

    # Trajectory dynamics (from trajectory.py)
    trajectory_path_length: Optional[float] = None
    trajectory_displacement: Optional[float] = None
    trajectory_tortuosity: Optional[float] = None
    trajectory_mean_velocity: Optional[float] = None

    # Trajectory integrity (from integrity.py)
    integrity_score: Optional[float] = None
    integrity_label: Optional[str] = None  # 'fragmented', 'living', 'rigid'
    integrity_autocorrelation: Optional[float] = None
    integrity_recurrence_rate: Optional[float] = None
    transformation_density: Optional[float] = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result['turn_states'] = [t.to_dict() for t in self.turn_states]
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class SessionAnalyzer:
    """
    Streaming analyzer for MASE dialogue sessions.

    Maintains state across turns for:
    - Basin history tracking
    - Rolling window metrics
    - Coherence pattern detection
    - Trajectory dynamics and integrity
    """

    def __init__(self, window_size: int = 5, trajectory_window: int = 50):
        """
        Initialize session analyzer.

        Args:
            window_size: Number of turns for rolling metrics
            trajectory_window: Maximum Ψ observations for trajectory buffer
        """
        self.window_size = window_size
        self.detector = BasinDetector()
        self.history = BasinHistory()

        # Trajectory and integrity analyzers
        self.trajectory = TrajectoryBuffer(window_size=trajectory_window)
        self.integrity_analyzer = IntegrityAnalyzer()
        self.transformation_detector = TransformationDetector()

        # Accumulated state
        self.embeddings: List[np.ndarray] = []
        self.texts: List[str] = []
        self.agents: List[str] = []
        self.turn_states: List[TurnState] = []
        self.window_metrics: List[dict] = []

    def reset(self) -> None:
        """Reset analyzer state for new session."""
        self.history.clear()
        self.trajectory.clear()
        self.embeddings = []
        self.texts = []
        self.agents = []
        self.turn_states = []
        self.window_metrics = []

    def process_turn(
        self,
        content: str,
        agent_id: str,
        embedding: np.ndarray = None
    ) -> TurnState:
        """
        Process a single turn and return state snapshot.

        Args:
            content: Turn text content
            agent_id: Agent ID
            embedding: Optional embedding vector

        Returns:
            TurnState for this turn
        """
        turn_number = len(self.texts)

        # Accumulate
        self.texts.append(content)
        self.agents.append(agent_id)
        if embedding is not None:
            self.embeddings.append(np.array(embedding))

        # Compute window metrics if enough data
        if len(self.embeddings) >= self.window_size:
            window_embs = np.array(self.embeddings[-self.window_size:])
            window_result = compute_metrics(window_embs)
            self.window_metrics.append({
                'delta_kappa': window_result.semantic_curvature,
                'delta_h': window_result.entropy_shift,
                'alpha': window_result.dfa_alpha
            })

        # Compute current metrics (use full history or window)
        if len(self.embeddings) >= 4:
            embs = np.array(self.embeddings)
            metrics_result = compute_metrics(embs)
            metrics = {
                'delta_kappa': metrics_result.semantic_curvature,
                'delta_h': metrics_result.entropy_shift,
                'alpha': metrics_result.dfa_alpha
            }
        else:
            metrics = {'delta_kappa': 0.0, 'delta_h': 0.0, 'alpha': 0.5}

        # Compute Psi vector
        psi = compute_psi_vector(
            metrics,
            turn_texts=self.texts,
            window_metrics=self.window_metrics if self.window_metrics else None
        )

        # Compute dialogue context
        ctx = compute_dialogue_context(
            turn_texts=self.texts,
            turn_agents=self.agents,
            window_metrics=self.window_metrics if self.window_metrics else None,
            embeddings=np.array(self.embeddings) if len(self.embeddings) >= 2 else None
        )

        # Detect basin
        basin, confidence, meta = self.detector.detect(
            psi_vector=psi,
            raw_metrics=metrics,
            dialogue_context=ctx,
            basin_history=self.history
        )

        # Record in history
        self.history.append(basin, confidence, turn=turn_number)

        # Track Ψ in trajectory buffer
        self.trajectory.append({
            'psi_semantic': psi['psi_semantic'],
            'psi_temporal': psi['psi_temporal'],
            'psi_affective': psi['psi_affective']
        })

        # Compute trajectory dynamics
        trajectory_deriv = compute_trajectory_derivatives(self.trajectory)

        # Create turn state
        state = TurnState(
            turn_number=turn_number,
            agent_id=agent_id,
            basin=basin,
            basin_confidence=confidence,
            psi_semantic=psi['psi_semantic'],
            psi_temporal=psi['psi_temporal'],
            psi_affective=psi['psi_affective'],
            coherence_pattern=ctx.coherence_pattern,
            residence_time=meta['residence_time'],
            velocity_magnitude=trajectory_deriv['speed'],
            acceleration_magnitude=self.trajectory.compute_acceleration_magnitude(),
            trajectory_curvature=trajectory_deriv['curvature']
        )
        self.turn_states.append(state)

        return state

    def get_summary(
        self,
        compute_ci: bool = False,
        bootstrap_iterations: int = 300
    ) -> SessionAnalysisResult:
        """
        Get summary analysis for accumulated session.

        Args:
            compute_ci: If True, compute bootstrap confidence intervals (slower)
            bootstrap_iterations: Number of bootstrap samples for CI

        Returns:
            SessionAnalysisResult with full analysis
        """
        # CI-related fields (populated if compute_ci=True)
        curvature_ci = None
        alpha_ci = None
        entropy_ci = None
        curvature_p_value = None
        alpha_r_squared = None
        entropy_stability = None
        curvature_significant = None
        alpha_in_range = None
        entropy_significant = None

        # Compute full-session metrics
        if len(self.embeddings) >= 4:
            embs = np.array(self.embeddings)

            if compute_ci:
                # Use enhanced metrics with CI
                metrics_ci = compute_metrics_with_ci(
                    embs,
                    bootstrap_iterations=bootstrap_iterations
                )
                sc = metrics_ci.semantic_curvature
                alpha = metrics_ci.dfa_alpha
                delta_h = metrics_ci.entropy_shift
                velocity_mean = metrics_ci.semantic_velocity_mean

                # Populate CI fields
                curvature_ci = metrics_ci.curvature_ci
                alpha_ci = metrics_ci.alpha_ci
                entropy_ci = metrics_ci.entropy_ci
                curvature_p_value = metrics_ci.curvature_p_value
                alpha_r_squared = metrics_ci.alpha_r_squared
                entropy_stability = metrics_ci.entropy_stability
                curvature_significant = metrics_ci.curvature_significant
                alpha_in_range = metrics_ci.alpha_in_range
                entropy_significant = metrics_ci.entropy_significant
            else:
                # Fast path: basic metrics only
                metrics = compute_metrics(embs)
                sc = metrics.semantic_curvature
                alpha = metrics.dfa_alpha
                delta_h = metrics.entropy_shift
                velocity_mean = metrics.semantic_velocity_mean
        else:
            sc = 0.0
            alpha = 0.5
            delta_h = 0.0
            velocity_mean = 0.0

        # Basin analysis
        basin_dist = self.history.get_basin_distribution()
        basin_seq = self.history.get_basin_sequence()

        if basin_dist:
            dominant = max(basin_dist, key=basin_dist.get)
            dominant_pct = basin_dist[dominant] / len(basin_seq)
        else:
            dominant = 'Transitional'
            dominant_pct = 0.0

        # Coherence pattern distribution
        coherence_dist = {}
        for state in self.turn_states:
            pattern = state.coherence_pattern
            coherence_dist[pattern] = coherence_dist.get(pattern, 0) + 1

        # Inquiry vs Mimicry ratio
        inquiry_count = basin_dist.get('Collaborative Inquiry', 0)
        mimicry_count = basin_dist.get('Cognitive Mimicry', 0)
        total_eval = inquiry_count + mimicry_count
        inquiry_ratio = inquiry_count / total_eval if total_eval > 0 else 0.5

        # Voice distinctiveness (final value)
        voice_dist = 0.0
        if len(self.embeddings) >= 2:
            ctx = compute_dialogue_context(
                self.texts, self.agents,
                embeddings=np.array(self.embeddings)
            )
            voice_dist = ctx.voice_distinctiveness

        # Affective analysis (using enhanced module)
        psi_affective = None
        sentiment_mean = None
        sentiment_variance = None
        hedging_density = None
        agent_sentiment = None
        agent_divergence = None

        if self.texts:
            affective_result = compute_affective_substrate(self.texts, self.agents)
            psi_affective = affective_result.psi_affective
            sentiment_mean = affective_result.sentiment_mean
            sentiment_variance = affective_result.sentiment_variance
            hedging_density = affective_result.hedging_density
            agent_sentiment = affective_result.agent_sentiment
            agent_divergence = compute_agent_affective_divergence(affective_result)

        # Trajectory dynamics
        trajectory_summary = self.trajectory.get_summary()
        trajectory_path_length = trajectory_summary['path_length']
        trajectory_displacement = trajectory_summary['displacement']
        trajectory_tortuosity = trajectory_summary['tortuosity']

        # Compute mean velocity from turn states
        velocities = [s.velocity_magnitude for s in self.turn_states if s.velocity_magnitude is not None]
        trajectory_mean_velocity = float(np.mean(velocities)) if velocities else None

        # Trajectory integrity
        integrity_result = self.integrity_analyzer.compute(self.trajectory, self.history)
        transformation_density = self.transformation_detector.compute_transformation_density(
            self.trajectory, self.history
        )

        return SessionAnalysisResult(
            semantic_curvature=sc,
            dfa_alpha=alpha,
            entropy_shift=delta_h,
            semantic_velocity_mean=velocity_mean,
            basin_sequence=basin_seq,
            basin_distribution=basin_dist,
            transition_count=self.history.get_transition_count(),
            dominant_basin=dominant,
            dominant_basin_percentage=dominant_pct,
            voice_distinctiveness=voice_dist,
            coherence_pattern_distribution=coherence_dist,
            inquiry_vs_mimicry_ratio=inquiry_ratio,
            turn_states=self.turn_states,
            n_turns=len(self.texts),
            agents=list(set(self.agents)),
            # CI fields (populated if compute_ci=True)
            curvature_ci=curvature_ci,
            alpha_ci=alpha_ci,
            entropy_ci=entropy_ci,
            curvature_p_value=curvature_p_value,
            alpha_r_squared=alpha_r_squared,
            entropy_stability=entropy_stability,
            curvature_significant=curvature_significant,
            alpha_in_range=alpha_in_range,
            entropy_significant=entropy_significant,
            # Affective analysis
            psi_affective=psi_affective,
            sentiment_mean=sentiment_mean,
            sentiment_variance=sentiment_variance,
            hedging_density=hedging_density,
            agent_sentiment=agent_sentiment,
            agent_affective_divergence=agent_divergence,
            # Trajectory dynamics
            trajectory_path_length=trajectory_path_length,
            trajectory_displacement=trajectory_displacement,
            trajectory_tortuosity=trajectory_tortuosity,
            trajectory_mean_velocity=trajectory_mean_velocity,
            # Trajectory integrity
            integrity_score=integrity_result.integrity_score,
            integrity_label=integrity_result.integrity_label,
            integrity_autocorrelation=integrity_result.autocorrelation,
            integrity_recurrence_rate=integrity_result.recurrence_rate,
            transformation_density=transformation_density
        )


def analyze_session(
    session_path: Path,
    compute_embeddings: bool = True,
    compute_ci: bool = False,
    bootstrap_iterations: int = 300
) -> SessionAnalysisResult:
    """
    Analyze a completed session from JSON file.

    Args:
        session_path: Path to session JSON
        compute_embeddings: If True, compute embeddings for turns that lack them.
            Uses sentence-transformers (all-mpnet-base-v2). Default True.
        compute_ci: If True, compute bootstrap confidence intervals (slower).
            Recommended for research-grade analysis.
        bootstrap_iterations: Number of bootstrap samples for CI computation.

    Returns:
        SessionAnalysisResult with full analysis (and CIs if compute_ci=True)
    """
    with open(session_path) as f:
        data = json.load(f)

    analyzer = SessionAnalyzer()

    # Lazy-load embedding service only if needed
    embedding_service = None

    for turn in data.get('turns', []):
        content = turn.get('content', '')
        agent_id = turn.get('agent_id', 'unknown')
        embedding = turn.get('embedding')

        if embedding is not None:
            embedding = np.array(embedding)
        elif compute_embeddings and content:
            # Compute embedding on the fly
            if embedding_service is None:
                try:
                    from .embedding_service import get_embedding_service
                except ImportError:
                    from embedding_service import get_embedding_service
                embedding_service = get_embedding_service()
            embedding = embedding_service.embed(content)

        analyzer.process_turn(content, agent_id, embedding)

    return analyzer.get_summary(
        compute_ci=compute_ci,
        bootstrap_iterations=bootstrap_iterations
    )


def compare_sessions(
    session_a_path: Path,
    session_b_path: Path,
    compute_ci: bool = False,
    bootstrap_iterations: int = 300
) -> Dict[str, Any]:
    """
    Compare two sessions for experimental analysis.

    Args:
        session_a_path: Path to first session
        session_b_path: Path to second session
        compute_ci: If True, compute bootstrap CIs and check for overlap
        bootstrap_iterations: Number of bootstrap samples

    Returns:
        Dict with comparison metrics, deltas, and (if compute_ci) CI overlap analysis
    """
    result_a = analyze_session(
        session_a_path,
        compute_ci=compute_ci,
        bootstrap_iterations=bootstrap_iterations
    )
    result_b = analyze_session(
        session_b_path,
        compute_ci=compute_ci,
        bootstrap_iterations=bootstrap_iterations
    )

    comparison = {
        'session_a': session_a_path.name,
        'session_b': session_b_path.name,

        # Metric deltas (B - A)
        'delta_semantic_curvature': result_b.semantic_curvature - result_a.semantic_curvature,
        'delta_dfa_alpha': result_b.dfa_alpha - result_a.dfa_alpha,
        'delta_entropy_shift': result_b.entropy_shift - result_a.entropy_shift,

        # Basin comparison
        'delta_transition_count': result_b.transition_count - result_a.transition_count,
        'delta_inquiry_ratio': result_b.inquiry_vs_mimicry_ratio - result_a.inquiry_vs_mimicry_ratio,
        'delta_voice_distinctiveness': result_b.voice_distinctiveness - result_a.voice_distinctiveness,

        # Dominant basins
        'dominant_basin_a': result_a.dominant_basin,
        'dominant_basin_b': result_b.dominant_basin,

        # Raw results for detailed analysis
        'result_a': result_a.to_dict(),
        'result_b': result_b.to_dict()
    }

    # Add CI overlap analysis if computed
    if compute_ci and result_a.alpha_ci and result_b.alpha_ci:
        def _ci_overlap(ci_a: Tuple[float, float], ci_b: Tuple[float, float]) -> bool:
            """Check if two CIs overlap (if they don't, difference may be significant)."""
            return not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])

        comparison['ci_analysis'] = {
            'curvature_ci_a': result_a.curvature_ci,
            'curvature_ci_b': result_b.curvature_ci,
            'curvature_cis_overlap': _ci_overlap(result_a.curvature_ci, result_b.curvature_ci),

            'alpha_ci_a': result_a.alpha_ci,
            'alpha_ci_b': result_b.alpha_ci,
            'alpha_cis_overlap': _ci_overlap(result_a.alpha_ci, result_b.alpha_ci),

            'entropy_ci_a': result_a.entropy_ci,
            'entropy_ci_b': result_b.entropy_ci,
            'entropy_cis_overlap': _ci_overlap(result_a.entropy_ci, result_b.entropy_ci),

            # R-squared for DFA fit quality
            'alpha_r_squared_a': result_a.alpha_r_squared,
            'alpha_r_squared_b': result_b.alpha_r_squared,

            # Threshold flags
            'curvature_significant_a': result_a.curvature_significant,
            'curvature_significant_b': result_b.curvature_significant,
            'alpha_in_range_a': result_a.alpha_in_range,
            'alpha_in_range_b': result_b.alpha_in_range,
            'entropy_significant_a': result_a.entropy_significant,
            'entropy_significant_b': result_b.entropy_significant,
        }

    return comparison


# Test if run directly
if __name__ == "__main__":
    import sys

    print("Session Analysis Module Test")
    print("=" * 50)

    # Check for session file argument
    if len(sys.argv) > 1:
        session_path = Path(sys.argv[1])
        if session_path.exists():
            print(f"\nAnalyzing: {session_path}")
            result = analyze_session(session_path)

            print(f"\n=== Core Metrics ===")
            print(f"  Semantic Curvature (Δκ): {result.semantic_curvature:.4f}")
            print(f"  DFA Alpha (α): {result.dfa_alpha:.4f}")
            print(f"  Entropy Shift (ΔH): {result.entropy_shift:.4f}")

            print(f"\n=== Basin Analysis ===")
            print(f"  Dominant Basin: {result.dominant_basin} ({result.dominant_basin_percentage:.1%})")
            print(f"  Transitions: {result.transition_count}")
            print(f"  Distribution: {result.basin_distribution}")

            print(f"\n=== Quality Indicators ===")
            print(f"  Inquiry vs Mimicry: {result.inquiry_vs_mimicry_ratio:.2f}")
            print(f"  Voice Distinctiveness: {result.voice_distinctiveness:.4f}")
            print(f"  Coherence Patterns: {result.coherence_pattern_distribution}")

            print(f"\n=== Trajectory Dynamics ===")
            print(f"  Path Length: {result.trajectory_path_length:.4f}" if result.trajectory_path_length else "  Path Length: N/A")
            print(f"  Displacement: {result.trajectory_displacement:.4f}" if result.trajectory_displacement else "  Displacement: N/A")
            print(f"  Tortuosity: {result.trajectory_tortuosity:.4f}" if result.trajectory_tortuosity else "  Tortuosity: N/A")
            print(f"  Mean Velocity: {result.trajectory_mean_velocity:.4f}" if result.trajectory_mean_velocity else "  Mean Velocity: N/A")

            print(f"\n=== Trajectory Integrity ===")
            print(f"  Integrity Score: {result.integrity_score:.3f}" if result.integrity_score else "  Integrity Score: N/A")
            print(f"  Integrity Label: {result.integrity_label}")
            print(f"  Autocorrelation: {result.integrity_autocorrelation:.3f}" if result.integrity_autocorrelation else "  Autocorrelation: N/A")
            print(f"  Recurrence Rate: {result.integrity_recurrence_rate:.3f}" if result.integrity_recurrence_rate else "  Recurrence Rate: N/A")
            print(f"  Transformation Density: {result.transformation_density:.3f}" if result.transformation_density else "  Transformation Density: N/A")
        else:
            print(f"File not found: {session_path}")
    else:
        print("\nUsage: python -m src.session_analysis <session.json>")
        print("\nRunning synthetic test...")

        # Synthetic test
        analyzer = SessionAnalyzer()

        # Simulate 10 turns
        np.random.seed(42)
        agents = ['ilya', 'elowen', 'orin', 'nyra', 'luma']

        for i in range(10):
            agent = agents[i % len(agents)]
            content = f"Turn {i} from {agent}: This is synthetic dialogue content for testing."
            embedding = np.random.randn(768)
            embedding /= np.linalg.norm(embedding)

            state = analyzer.process_turn(content, agent, embedding)
            print(f"  Turn {i}: {agent} -> {state.basin} (conf={state.basin_confidence:.2f})")

        summary = analyzer.get_summary()
        print(f"\nSummary:")
        print(f"  Dominant Basin: {summary.dominant_basin}")
        print(f"  Transitions: {summary.transition_count}")
        print(f"  Inquiry Ratio: {summary.inquiry_vs_mimicry_ratio:.2f}")

        print(f"\n  --- Trajectory ---")
        print(f"  Path Length: {summary.trajectory_path_length:.4f}" if summary.trajectory_path_length else "  Path Length: N/A")
        print(f"  Tortuosity: {summary.trajectory_tortuosity:.4f}" if summary.trajectory_tortuosity else "  Tortuosity: N/A")

        print(f"\n  --- Integrity ---")
        print(f"  Score: {summary.integrity_score:.3f} ({summary.integrity_label})")
