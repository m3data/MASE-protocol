"""
Attractor Basin Detection for MASE Multi-Agent Dialogue.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Ported from Semantic Climate Phase Space with adaptations for
multi-agent dialogue (no biosignal substrate, agent-specific context).

Basin Taxonomy (adapted for MASE):
    1. Deep Resonance: High semantic engagement + high affect
    2. Collaborative Inquiry: Genuine co-exploration with uncertainty
    3. Cognitive Mimicry: Performed engagement without genuine uncertainty
    4. Reflexive Performance: Performed self-examination (scripted)
    5. Sycophantic Convergence: High agreement + low exploration
    6. Creative Dilation: Divergent exploration + high affect
    7. Generative Conflict: Productive tension driving exploration
    8. Dissociation: Low engagement across substrates
    9. Transitional: Between stable basins

Note: Embodied Coherence removed (requires biosignal data).
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import numpy as np
import re

__all__ = [
    'BasinHistory',
    'BasinDetector',
    'DialogueContext',
    'compute_psi_vector',
    'compute_affective_substrate',
    'compute_dialogue_context',
]


@dataclass
class DialogueContext:
    """
    Context features for refined basin classification.

    Distinguishes performative from genuine engagement in multi-agent dialogue.

    Attributes:
        hedging_density: Proportion of uncertainty markers
        turn_length_variance: Variance in response lengths across agents
        delta_kappa_variance: Responsiveness of trajectory (scripted vs adaptive)
        voice_distinctiveness: How different agents sound from each other
        coherence_pattern: 'breathing', 'locked', 'fragmented', 'transitional'
    """
    hedging_density: float = 0.0
    turn_length_variance: float = 0.0
    delta_kappa_variance: float = 0.0
    voice_distinctiveness: float = 0.0
    coherence_pattern: str = 'transitional'

    def to_dict(self) -> dict:
        return asdict(self)


class BasinHistory:
    """
    Tracks basin sequence for hysteresis-aware detection.

    Enables:
    - Residence time computation
    - Transition counting
    - Basin sequence analysis
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history: List[Tuple[int, str, float]] = []  # (turn, basin, confidence)
        self._current_basin: Optional[str] = None
        self._previous_basin: Optional[str] = None
        self._basin_entry_turn: int = 0
        self._transition_count: int = 0

    def append(self, basin: str, confidence: float, turn: int = None) -> None:
        """Add a basin entry to history."""
        if turn is None:
            turn = len(self.history)

        if self._current_basin is not None and basin != self._current_basin:
            self._previous_basin = self._current_basin
            self._basin_entry_turn = turn
            self._transition_count += 1

        if self._current_basin is None:
            self._basin_entry_turn = turn

        self._current_basin = basin
        self.history.append((turn, basin, confidence))

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_current_basin(self) -> Optional[str]:
        return self._current_basin

    def get_residence_time(self) -> int:
        """Consecutive turns in current basin."""
        if not self.history:
            return 0

        current = self._current_basin
        count = 0
        for _, basin, _ in reversed(self.history):
            if basin == current:
                count += 1
            else:
                break
        return count

    def get_previous_basin(self) -> Optional[str]:
        return self._previous_basin

    def get_transition_count(self) -> int:
        return self._transition_count

    def get_basin_sequence(self, n: int = None) -> List[str]:
        """Get sequence of recent basins."""
        if n is None:
            return [basin for _, basin, _ in self.history]
        return [basin for _, basin, _ in self.history[-n:]]

    def get_basin_distribution(self) -> Dict[str, int]:
        """Count visits to each basin."""
        dist = defaultdict(int)
        for _, basin, _ in self.history:
            dist[basin] += 1
        return dict(dist)

    def get_transition_matrix(self) -> Dict[Tuple[str, str], int]:
        """Count transitions between basins."""
        matrix = defaultdict(int)
        for i in range(1, len(self.history)):
            prev_basin = self.history[i-1][1]
            curr_basin = self.history[i][1]
            if prev_basin != curr_basin:
                matrix[(prev_basin, curr_basin)] += 1
        return dict(matrix)

    def clear(self) -> None:
        self.history = []
        self._current_basin = None
        self._previous_basin = None
        self._basin_entry_turn = 0
        self._transition_count = 0


class BasinDetector:
    """
    Detects attractor basins in multi-agent dialogue.

    Adapted from Semantic Climate for MASE context:
    - No biosignal substrate (all digital)
    - Multi-agent turn dynamics instead of human-AI
    - Voice distinctiveness as quality signal
    """

    # Canonical basins for multi-agent dialogue
    BASINS = [
        'Deep Resonance',
        'Collaborative Inquiry',
        'Cognitive Mimicry',
        'Reflexive Performance',
        'Sycophantic Convergence',
        'Creative Dilation',
        'Generative Conflict',
        'Dissociation',
        'Transitional'
    ]

    def __init__(
        self,
        residence_confidence_modulation: bool = True,
        new_entry_penalty: float = 0.7,
        settled_bonus: float = 1.1,
        settled_threshold: int = 5
    ):
        self.residence_confidence_modulation = residence_confidence_modulation
        self.new_entry_penalty = new_entry_penalty
        self.settled_bonus = settled_bonus
        self.settled_threshold = settled_threshold

    def detect(
        self,
        psi_vector: dict,
        raw_metrics: dict = None,
        dialogue_context: DialogueContext = None,
        basin_history: BasinHistory = None
    ) -> Tuple[str, float, dict]:
        """
        Classify dialogue state into attractor basin.

        Args:
            psi_vector: Dict with psi_semantic, psi_temporal, psi_affective
            raw_metrics: Dict with delta_kappa, delta_h, alpha
            dialogue_context: DialogueContext for refined classification
            basin_history: BasinHistory for hysteresis

        Returns:
            (basin_name, confidence, metadata)
        """
        basin_name, raw_confidence = self._classify_basin(
            psi_vector, raw_metrics, dialogue_context
        )

        metadata = {
            'raw_confidence': raw_confidence,
            'residence_time': 0,
            'previous_basin': None
        }

        final_confidence = raw_confidence

        if basin_history is not None:
            metadata['residence_time'] = basin_history.get_residence_time()
            metadata['previous_basin'] = basin_history.get_previous_basin()

            if self.residence_confidence_modulation:
                current_basin = basin_history.get_current_basin()

                if current_basin is None:
                    final_confidence = raw_confidence * self.new_entry_penalty
                elif basin_name != current_basin:
                    final_confidence = raw_confidence * self.new_entry_penalty
                elif metadata['residence_time'] >= self.settled_threshold:
                    final_confidence = min(1.0, raw_confidence * self.settled_bonus)

        return (basin_name, float(final_confidence), metadata)

    def _classify_basin(
        self,
        psi_vector: dict,
        raw_metrics: dict = None,
        dialogue_context: DialogueContext = None
    ) -> Tuple[str, float]:
        """
        Core basin classification logic for multi-agent dialogue.
        """
        sem = psi_vector.get('psi_semantic', 0.0) or 0.0
        temp = psi_vector.get('psi_temporal', 0.5) or 0.5
        aff = psi_vector.get('psi_affective', 0.0) or 0.0

        delta_kappa = raw_metrics.get('delta_kappa', 0.0) if raw_metrics else 0.0

        # Extract dialogue context
        ctx = dialogue_context or DialogueContext()
        hedging = ctx.hedging_density
        voice_dist = ctx.voice_distinctiveness
        dk_variance = ctx.delta_kappa_variance
        coherence = ctx.coherence_pattern

        # === HIGH-CERTAINTY BASINS ===

        # Deep Resonance: All substrates high, good voice distinctiveness
        if sem > 0.4 and aff > 0.4 and voice_dist > 0.3:
            confidence = min(abs(sem), abs(aff), voice_dist)
            return ("Deep Resonance", float(confidence))

        # Dissociation: All substrates low
        if abs(sem) < 0.2 and abs(aff) < 0.2:
            confidence = 1.0 - max(abs(sem), abs(aff))
            return ("Dissociation", float(confidence))

        # === HIGH-AFFECT BASINS ===

        # Generative Conflict: Productive tension
        if abs(sem) > 0.3 and delta_kappa > 0.35 and aff > 0.3:
            confidence = min((delta_kappa / 0.7), abs(aff))
            return ("Generative Conflict", float(confidence))

        # Creative Dilation: Expansive exploration with feeling
        if delta_kappa > 0.35 and aff > 0.3:
            confidence = (delta_kappa / 0.7) * abs(aff)
            return ("Creative Dilation", float(confidence))

        # Sycophantic Convergence: High agreement, low exploration
        if sem > 0.3 and delta_kappa < 0.35 and aff < 0.2 and voice_dist < 0.3:
            confidence = sem * (1.0 - delta_kappa / 0.35) * (1.0 - voice_dist)
            return ("Sycophantic Convergence", float(confidence))

        # === SEMANTIC-ACTIVE, LOW-AFFECT TERRITORY ===
        # Distinguish mimicry from genuine inquiry

        if abs(sem) > 0.3 and aff < 0.2:
            # Collaborative Inquiry indicators:
            # - Hedging present (genuine uncertainty)
            # - Good voice distinctiveness (agents maintain identity)
            # - Responsive trajectory (dk_variance)
            # - Breathing coherence pattern
            inquiry_score = 0.0
            if hedging > 0.02:
                inquiry_score += 0.3
            if voice_dist > 0.3:
                inquiry_score += 0.3
            if dk_variance > 0.01:
                inquiry_score += 0.2
            if coherence == 'breathing':
                inquiry_score += 0.2

            # Cognitive Mimicry indicators:
            # - Low hedging (confident performance)
            # - Low voice distinctiveness (agents blur)
            # - Smooth trajectory (scripted)
            # - Locked coherence
            mimicry_score = 0.0
            if hedging < 0.01:
                mimicry_score += 0.3
            if voice_dist < 0.2:
                mimicry_score += 0.3
            if dk_variance < 0.005:
                mimicry_score += 0.2
            if coherence in ('locked', 'transitional'):
                mimicry_score += 0.2

            # Reflexive Performance indicators:
            # - Moderate hedging (performed uncertainty)
            # - Medium dk_variance (scripted oscillation)
            reflexive_score = 0.0
            if 0.01 <= hedging <= 0.03:
                reflexive_score += 0.3
            if 0.005 <= dk_variance <= 0.015:
                reflexive_score += 0.3
            if coherence == 'transitional':
                reflexive_score += 0.2
            if 0.2 <= voice_dist <= 0.4:
                reflexive_score += 0.2

            scores = {
                'Collaborative Inquiry': inquiry_score,
                'Cognitive Mimicry': mimicry_score,
                'Reflexive Performance': reflexive_score
            }
            best_basin = max(scores, key=scores.get)
            best_score = scores[best_basin]

            sorted_scores = sorted(scores.values(), reverse=True)
            if sorted_scores[0] - sorted_scores[1] < 0.1:
                confidence = abs(sem) * 0.5
            else:
                confidence = abs(sem) * (0.5 + best_score * 0.5)

            return (best_basin, float(confidence))

        # === DEFAULT: Transitional ===
        magnitudes = {
            'semantic': abs(sem),
            'affective': abs(aff),
            'temporal': abs(temp - 0.5)
        }
        dominant = max(magnitudes, key=magnitudes.get)
        confidence = magnitudes[dominant]

        if dominant == 'semantic' and delta_kappa > 0.35:
            return ("Creative Dilation", float(confidence))
        elif dominant == 'affective':
            return ("Generative Conflict" if delta_kappa > 0.35 else "Cognitive Mimicry", float(confidence))
        else:
            return ("Transitional", 0.3)


def compute_affective_substrate(turn_texts: List[str]) -> dict:
    """
    Calculate psi_affective from dialogue turn texts using VADER.

    Adapted for multi-agent dialogue (no human-AI distinction).

    Args:
        turn_texts: List of turn content strings

    Returns:
        dict with psi_affective, hedging_density, sentiment_trajectory
    """
    empty_result = {
        'psi_affective': 0.0,
        'sentiment_trajectory': [],
        'hedging_density': 0.0,
        'vulnerability_score': 0.0
    }

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return empty_result

    if not turn_texts:
        return empty_result

    vader = SentimentIntensityAnalyzer()

    # Sentiment trajectory
    sentiment_scores = []
    for text in turn_texts:
        scores = vader.polarity_scores(text)
        sentiment_scores.append(scores['compound'])

    sentiment_variance = float(np.var(sentiment_scores)) if len(sentiment_scores) > 1 else 0.0

    # Hedging patterns (uncertainty markers)
    hedging_patterns = [
        r'\b(I think|I guess|maybe|perhaps|possibly|probably|might|could be|seems like|sort of|kind of)\b',
        r'\b(I\'m not sure|I wonder|I feel like|it appears|it seems)\b',
        r'\b(arguably|presumably|apparently|seemingly)\b'
    ]

    total_words = 0
    hedging_count = 0
    for text in turn_texts:
        words = text.split()
        total_words += len(words)
        for pattern in hedging_patterns:
            hedging_count += len(re.findall(pattern, text, re.IGNORECASE))

    hedging_density = float(hedging_count / max(total_words, 1))

    # Vulnerability/openness indicators
    vulnerability_patterns = [
        r'\b(I feel|I\'m feeling|I felt)\b',
        r'\b(honestly|to be honest|truthfully)\b',
        r'\b(I don\'t know|I\'m not sure|I\'m uncertain)\b'
    ]

    vulnerability_count = 0
    for text in turn_texts:
        for pattern in vulnerability_patterns:
            vulnerability_count += len(re.findall(pattern, text, re.IGNORECASE))

    vulnerability_score = float(vulnerability_count / max(total_words, 1))

    # Composite psi_affective
    sentiment_norm = min(sentiment_variance / 0.5, 1.0)
    hedging_norm = min(hedging_density / 0.1, 1.0)
    vulnerability_norm = min(vulnerability_score / 0.05, 1.0)

    psi_affective_raw = (
        0.4 * sentiment_norm +
        0.3 * hedging_norm +
        0.3 * vulnerability_norm
    )
    psi_affective = np.tanh(2 * (psi_affective_raw - 0.5))

    return {
        'psi_affective': float(psi_affective),
        'sentiment_trajectory': sentiment_scores,
        'hedging_density': float(hedging_density),
        'vulnerability_score': float(vulnerability_score)
    }


def compute_psi_vector(
    metrics: dict,
    turn_texts: List[str] = None,
    window_metrics: List[dict] = None
) -> dict:
    """
    Compute Psi vector from metrics and turn texts.

    Args:
        metrics: Dict with delta_kappa, delta_h (or entropy_shift), alpha (or dfa_alpha)
        turn_texts: List of turn content strings for affective analysis
        window_metrics: List of per-window metrics for temporal stability

    Returns:
        dict with psi_semantic, psi_temporal, psi_affective
    """
    # Normalize metric keys
    delta_kappa = metrics.get('delta_kappa') or metrics.get('semantic_curvature', 0.0)
    delta_h = metrics.get('delta_h') or metrics.get('entropy_shift', 0.0)
    alpha = metrics.get('alpha') or metrics.get('dfa_alpha', 0.5)

    # Psi_semantic from core metrics (PC1 approximation)
    weights = np.array([0.577, 0.577, 0.577])
    metric_std = np.array([
        (delta_kappa - 0.15) / 0.15,
        (delta_h - 0.15) / 0.15,
        (alpha - 0.8) / 0.3
    ])
    psi_semantic_raw = np.dot(metric_std, weights) / np.linalg.norm(weights)
    psi_semantic = np.tanh(psi_semantic_raw)

    # Psi_temporal from metric stability
    psi_temporal = 0.5
    if window_metrics and len(window_metrics) >= 3:
        dk_values = [m.get('delta_kappa', 0) for m in window_metrics if m]
        if len(dk_values) >= 3:
            cv = np.std(dk_values) / (np.abs(np.mean(dk_values)) + 1e-10)
            psi_temporal = 1 / (1 + cv)

    # Psi_affective from text analysis
    psi_affective = 0.0
    hedging_density = 0.0
    if turn_texts:
        affective_result = compute_affective_substrate(turn_texts)
        psi_affective = affective_result['psi_affective']
        hedging_density = affective_result['hedging_density']

    return {
        'psi_semantic': float(psi_semantic),
        'psi_temporal': float(psi_temporal),
        'psi_affective': float(psi_affective),
        'hedging_density': hedging_density,
        'raw_metrics': {
            'delta_kappa': delta_kappa,
            'delta_h': delta_h,
            'alpha': alpha
        }
    }


def compute_dialogue_context(
    turn_texts: List[str],
    turn_agents: List[str],
    window_metrics: List[dict] = None,
    embeddings: np.ndarray = None
) -> DialogueContext:
    """
    Compute dialogue context for refined basin classification.

    Args:
        turn_texts: List of turn content strings
        turn_agents: List of agent IDs per turn
        window_metrics: List of per-window metric dicts
        embeddings: Turn embeddings for voice distinctiveness

    Returns:
        DialogueContext with features for basin classification
    """
    ctx = DialogueContext()

    if not turn_texts:
        return ctx

    # Hedging density
    affective = compute_affective_substrate(turn_texts)
    ctx.hedging_density = affective['hedging_density']

    # Turn length variance across agents
    if turn_texts and turn_agents and len(turn_texts) == len(turn_agents):
        agent_lengths = defaultdict(list)
        for text, agent in zip(turn_texts, turn_agents):
            agent_lengths[agent].append(len(text.split()))

        if len(agent_lengths) >= 2:
            agent_means = [np.mean(lengths) for lengths in agent_lengths.values()]
            ctx.turn_length_variance = float(np.var(agent_means))

    # Delta kappa variance (trajectory responsiveness)
    if window_metrics and len(window_metrics) >= 2:
        dk_values = [
            m.get('delta_kappa', 0) or m.get('semantic_curvature', 0)
            for m in window_metrics if m
        ]
        if len(dk_values) >= 2:
            ctx.delta_kappa_variance = float(np.var(dk_values))

    # Voice distinctiveness from embeddings
    if embeddings is not None and turn_agents and len(embeddings) == len(turn_agents):
        agent_embeddings = defaultdict(list)
        for emb, agent in zip(embeddings, turn_agents):
            if emb is not None:
                agent_embeddings[agent].append(emb)

        if len(agent_embeddings) >= 2:
            # Compute mean embedding per agent
            agent_centroids = {}
            for agent, embs in agent_embeddings.items():
                if embs:
                    agent_centroids[agent] = np.mean(embs, axis=0)

            # Average pairwise distance between agent centroids
            if len(agent_centroids) >= 2:
                centroids = list(agent_centroids.values())
                distances = []
                for i in range(len(centroids)):
                    for j in range(i + 1, len(centroids)):
                        d = 1 - np.dot(centroids[i], centroids[j]) / (
                            np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j]) + 1e-10
                        )
                        distances.append(d)
                ctx.voice_distinctiveness = float(np.mean(distances)) if distances else 0.0

    # Coherence pattern from autocorrelation of semantic velocity
    if embeddings is not None and len(embeddings) >= 6:
        try:
            from .metrics import semantic_velocity
        except ImportError:
            from metrics import semantic_velocity
        velocity = semantic_velocity(embeddings)
        if len(velocity) >= 5:
            # Autocorrelation at lag 1
            autocorr = np.corrcoef(velocity[:-1], velocity[1:])[0, 1]
            if np.isnan(autocorr):
                ctx.coherence_pattern = 'transitional'
            elif autocorr < -0.2:
                ctx.coherence_pattern = 'breathing'  # Negative autocorr = explore/consolidate rhythm
            elif autocorr > 0.3:
                ctx.coherence_pattern = 'locked'  # Positive autocorr = stuck
            elif np.var(velocity) > 0.1:
                ctx.coherence_pattern = 'fragmented'  # High variance = chaotic
            else:
                ctx.coherence_pattern = 'transitional'

    return ctx


# Test if run directly
if __name__ == "__main__":
    print("Basin Detection Module Test")
    print("=" * 50)

    # Synthetic test
    detector = BasinDetector()
    history = BasinHistory()

    # Simulate progression through basins
    test_states = [
        {'psi_semantic': 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.1},  # Transitional
        {'psi_semantic': 0.4, 'psi_temporal': 0.6, 'psi_affective': 0.1},  # Cognitive Mimicry territory
        {'psi_semantic': 0.5, 'psi_temporal': 0.7, 'psi_affective': 0.4},  # Creative Dilation
        {'psi_semantic': 0.6, 'psi_temporal': 0.8, 'psi_affective': 0.5},  # Deep Resonance
    ]

    raw_metrics = {'delta_kappa': 0.4, 'delta_h': 0.15, 'alpha': 0.85}
    ctx = DialogueContext(hedging_density=0.03, voice_distinctiveness=0.4)

    print("\nBasin Detection Sequence:")
    for i, psi in enumerate(test_states):
        basin, confidence, meta = detector.detect(psi, raw_metrics, ctx, history)
        history.append(basin, confidence, turn=i)
        print(f"  Turn {i}: {basin} (conf={confidence:.2f}, residence={meta['residence_time']})")

    print(f"\nBasin Distribution: {history.get_basin_distribution()}")
    print(f"Transitions: {history.get_transition_count()}")
