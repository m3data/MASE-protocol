"""
Session Analysis Module for MASE.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Provides post-hoc and streaming analysis of MASE dialogue sessions,
integrating metrics computation with basin detection.

Usage:
    from src.session_analysis import analyze_session, SessionAnalyzer

    # Quick analysis of a session file
    results = analyze_session(Path("experiments/runs/session.json"))

    # Streaming analysis during dialogue
    analyzer = SessionAnalyzer()
    for turn in dialogue:
        state = analyzer.process_turn(turn)
        print(f"Basin: {state['basin']}")
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

from .metrics import (
    compute_metrics,
    semantic_curvature,
    dfa_alpha,
    entropy_shift,
    semantic_velocity
)
from .basins import (
    BasinDetector,
    BasinHistory,
    DialogueContext,
    compute_psi_vector,
    compute_dialogue_context
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
    """

    def __init__(self, window_size: int = 5):
        """
        Initialize session analyzer.

        Args:
            window_size: Number of turns for rolling metrics
        """
        self.window_size = window_size
        self.detector = BasinDetector()
        self.history = BasinHistory()

        # Accumulated state
        self.embeddings: List[np.ndarray] = []
        self.texts: List[str] = []
        self.agents: List[str] = []
        self.turn_states: List[TurnState] = []
        self.window_metrics: List[dict] = []

    def reset(self) -> None:
        """Reset analyzer state for new session."""
        self.history.clear()
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
            residence_time=meta['residence_time']
        )
        self.turn_states.append(state)

        return state

    def get_summary(self) -> SessionAnalysisResult:
        """
        Get summary analysis for accumulated session.

        Returns:
            SessionAnalysisResult with full analysis
        """
        # Compute full-session metrics
        if len(self.embeddings) >= 4:
            embs = np.array(self.embeddings)
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
            agents=list(set(self.agents))
        )


def analyze_session(session_path: Path) -> SessionAnalysisResult:
    """
    Analyze a completed session from JSON file.

    Args:
        session_path: Path to session JSON

    Returns:
        SessionAnalysisResult with full analysis
    """
    with open(session_path) as f:
        data = json.load(f)

    analyzer = SessionAnalyzer()

    for turn in data.get('turns', []):
        content = turn.get('content', '')
        agent_id = turn.get('agent_id', 'unknown')
        embedding = turn.get('embedding')

        if embedding is not None:
            embedding = np.array(embedding)

        analyzer.process_turn(content, agent_id, embedding)

    return analyzer.get_summary()


def compare_sessions(
    session_a_path: Path,
    session_b_path: Path
) -> Dict[str, Any]:
    """
    Compare two sessions for experimental analysis.

    Args:
        session_a_path: Path to first session
        session_b_path: Path to second session

    Returns:
        Dict with comparison metrics and deltas
    """
    result_a = analyze_session(session_a_path)
    result_b = analyze_session(session_b_path)

    return {
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
