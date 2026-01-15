"""
Affective Substrate Analysis for MASE.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Ported from Semantic Climate Phase Space. Provides sentiment and
affective pattern analysis for multi-agent dialogues.

Key features:
- VADER-based sentiment analysis (fast, lexicon-based)
- Hedging pattern detection (epistemic uncertainty markers)
- Vulnerability indicators (emotional openness signals)
- Confidence markers (assertion strength)
- Agent-specific sentiment trajectories

Usage:
    from src.affective import compute_affective_substrate, AffectiveResult

    result = compute_affective_substrate(turn_texts, turn_agents)
    print(f"Ψ_affective: {result.psi_affective}")
    print(f"Agent sentiment: {result.agent_sentiment}")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import re


@dataclass
class AffectiveResult:
    """Results from affective substrate analysis."""
    # Composite score
    psi_affective: float  # [-1, 1] affective substrate value

    # Sentiment analysis
    sentiment_trajectory: List[float]  # Per-turn VADER compound scores
    sentiment_mean: float
    sentiment_variance: float

    # Pattern densities
    hedging_density: float       # Epistemic uncertainty markers
    vulnerability_score: float   # Emotional openness signals
    confidence_variance: float   # Assertion strength variability

    # Agent-specific (unique to MASE)
    agent_sentiment: Dict[str, float]  # Mean sentiment per agent
    agent_hedging: Dict[str, float]    # Hedging density per agent

    # Metadata
    n_turns: int
    source: str = "vader"

    def to_dict(self) -> dict:
        return {
            'psi_affective': self.psi_affective,
            'sentiment_trajectory': self.sentiment_trajectory,
            'sentiment_mean': self.sentiment_mean,
            'sentiment_variance': self.sentiment_variance,
            'hedging_density': self.hedging_density,
            'vulnerability_score': self.vulnerability_score,
            'confidence_variance': self.confidence_variance,
            'agent_sentiment': self.agent_sentiment,
            'agent_hedging': self.agent_hedging,
            'n_turns': self.n_turns,
            'source': self.source
        }


# Regex patterns for affective signals
HEDGING_PATTERNS = [
    r'\b(I think|I guess|I suppose|maybe|perhaps|possibly|probably|might|could be|seems like|sort of|kind of)\b',
    r'\b(I\'m not sure|I wonder|I feel like|it appears|it seems)\b',
    r'\b(arguably|presumably|apparently|seemingly)\b'
]

VULNERABILITY_PATTERNS = [
    r'\b(I feel|I\'m feeling|I felt)\b',
    r'\b(I\'m|I am)\s+(scared|worried|afraid|anxious|nervous|uncertain|confused|overwhelmed)\b',
    r'\b(my|I)\s+(fear|worry|concern|anxiety|doubt)\b',
    r'\b(honestly|to be honest|truthfully|frankly)\b',
    r'\b(I don\'t know|I\'m struggling|I\'m not sure|I\'m uncertain)\b'
]

CONFIDENCE_PATTERNS = [
    r'\b(definitely|certainly|absolutely|clearly|obviously|undoubtedly)\b',
    r'\b(I\'m certain|I\'m sure|I know|without doubt|no question)\b',
    r'\b(always|never|must|will)\b'
]

EMOTION_WORDS = [
    'afraid', 'angry', 'anxious', 'confused', 'disappointed', 'excited',
    'frustrated', 'grateful', 'happy', 'hopeful', 'lonely', 'sad',
    'scared', 'surprised', 'uncertain', 'worried'
]


def _count_patterns(text: str, patterns: List[str]) -> int:
    """Count regex pattern matches in text."""
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def _count_emotion_words(text: str) -> int:
    """Count emotion word occurrences."""
    text_lower = text.lower()
    count = 0
    for word in EMOTION_WORDS:
        if re.search(r'\b' + word + r'\b', text_lower):
            count += 1
    return count


def compute_affective_substrate(
    turn_texts: List[str],
    turn_agents: List[str] = None
) -> AffectiveResult:
    """
    Compute affective substrate from dialogue turns.

    Uses VADER lexicon-based sentiment analysis plus pattern detection
    for hedging, vulnerability, and confidence markers.

    Args:
        turn_texts: List of turn text content
        turn_agents: Optional list of agent IDs per turn

    Returns:
        AffectiveResult with sentiment and pattern analysis
    """
    # Try to import VADER
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
        has_vader = True
    except ImportError:
        has_vader = False

    if not turn_texts:
        return AffectiveResult(
            psi_affective=0.0,
            sentiment_trajectory=[],
            sentiment_mean=0.0,
            sentiment_variance=0.0,
            hedging_density=0.0,
            vulnerability_score=0.0,
            confidence_variance=0.0,
            agent_sentiment={},
            agent_hedging={},
            n_turns=0
        )

    n_turns = len(turn_texts)

    # Initialize agent tracking
    if turn_agents is None:
        turn_agents = ['unknown'] * n_turns

    agent_sentiments: Dict[str, List[float]] = {}
    agent_hedging_counts: Dict[str, Tuple[int, int]] = {}  # (count, words)

    # Per-turn analysis
    sentiment_scores = []
    confidence_densities = []
    total_words = 0
    total_hedging = 0
    total_vulnerability = 0

    for i, text in enumerate(turn_texts):
        agent = turn_agents[i] if i < len(turn_agents) else 'unknown'
        words = text.split()
        word_count = len(words)
        total_words += word_count

        # Sentiment
        if has_vader:
            scores = vader.polarity_scores(text)
            compound = scores['compound']
        else:
            # Fallback: neutral
            compound = 0.0

        sentiment_scores.append(compound)

        # Track per-agent sentiment
        if agent not in agent_sentiments:
            agent_sentiments[agent] = []
        agent_sentiments[agent].append(compound)

        # Hedging
        hedging_count = _count_patterns(text, HEDGING_PATTERNS)
        total_hedging += hedging_count

        if agent not in agent_hedging_counts:
            agent_hedging_counts[agent] = (0, 0)
        prev = agent_hedging_counts[agent]
        agent_hedging_counts[agent] = (prev[0] + hedging_count, prev[1] + word_count)

        # Vulnerability
        vuln_patterns = _count_patterns(text, VULNERABILITY_PATTERNS)
        vuln_emotions = _count_emotion_words(text)
        total_vulnerability += vuln_patterns + vuln_emotions

        # Confidence (per-turn density for variance)
        conf_count = _count_patterns(text, CONFIDENCE_PATTERNS)
        confidence_densities.append(conf_count / max(word_count, 1))

    # Aggregate metrics
    sentiment_mean = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
    sentiment_variance = float(np.var(sentiment_scores)) if len(sentiment_scores) > 1 else 0.0
    hedging_density = total_hedging / max(total_words, 1)
    vulnerability_score = total_vulnerability / max(total_words, 1)
    confidence_variance = float(np.var(confidence_densities)) if len(confidence_densities) > 1 else 0.0

    # Agent-level aggregates
    agent_sentiment = {
        agent: float(np.mean(scores))
        for agent, scores in agent_sentiments.items()
    }
    agent_hedging = {
        agent: counts[0] / max(counts[1], 1)
        for agent, counts in agent_hedging_counts.items()
    }

    # Composite Ψ_affective
    # Normalize components to [0, 1] range
    sentiment_norm = min(sentiment_variance / 0.5, 1.0)
    hedging_norm = min(hedging_density / 0.1, 1.0)
    vulnerability_norm = min(vulnerability_score / 0.05, 1.0)
    confidence_norm = min(confidence_variance / 0.01, 1.0)

    # Weighted combination
    psi_raw = (
        0.3 * sentiment_norm +
        0.3 * hedging_norm +
        0.3 * vulnerability_norm +
        0.1 * confidence_norm
    )

    # Map to [-1, 1] via tanh
    psi_affective = float(np.tanh(2 * (psi_raw - 0.5)))

    return AffectiveResult(
        psi_affective=psi_affective,
        sentiment_trajectory=sentiment_scores,
        sentiment_mean=sentiment_mean,
        sentiment_variance=sentiment_variance,
        hedging_density=hedging_density,
        vulnerability_score=vulnerability_score,
        confidence_variance=confidence_variance,
        agent_sentiment=agent_sentiment,
        agent_hedging=agent_hedging,
        n_turns=n_turns,
        source="vader" if has_vader else "fallback"
    )


def compute_agent_affective_divergence(result: AffectiveResult) -> float:
    """
    Compute divergence in affective patterns across agents.

    High divergence suggests agents have distinct emotional signatures.
    Low divergence suggests affective homogeneity (potential echo chamber).

    Args:
        result: AffectiveResult from compute_affective_substrate

    Returns:
        float: Divergence score [0, 1]
    """
    if len(result.agent_sentiment) < 2:
        return 0.0

    sentiments = list(result.agent_sentiment.values())
    hedging = list(result.agent_hedging.values())

    # Compute coefficient of variation for both
    sent_cv = np.std(sentiments) / (np.abs(np.mean(sentiments)) + 1e-10)
    hedge_cv = np.std(hedging) / (np.abs(np.mean(hedging)) + 1e-10)

    # Normalize and combine
    divergence = (np.tanh(sent_cv) + np.tanh(hedge_cv)) / 2

    return float(divergence)


# Test if run directly
if __name__ == "__main__":
    print("Affective Substrate Module Test")
    print("=" * 50)

    # Synthetic multi-agent dialogue
    turns = [
        "I think this is an interesting question, but I'm not sure about the answer.",
        "Perhaps we should consider multiple perspectives here.",
        "I feel uncertain about this approach. It seems risky.",
        "Definitely! This is clearly the right way forward.",
        "I wonder if there's something we're missing...",
        "Honestly, I'm a bit worried about the implications.",
        "This is absolutely fascinating. Without doubt, we should explore further.",
        "Maybe, but I'm struggling to see the full picture.",
    ]

    agents = ['ilya', 'elowen', 'orin', 'tala', 'nyra', 'luma', 'sefi', 'ilya']

    print(f"\nAnalyzing {len(turns)} turns from {len(set(agents))} agents...")

    result = compute_affective_substrate(turns, agents)

    print(f"\n=== Affective Substrate ===")
    print(f"Ψ_affective: {result.psi_affective:.4f}")
    print(f"Source: {result.source}")

    print(f"\n=== Sentiment ===")
    print(f"Mean: {result.sentiment_mean:.4f}")
    print(f"Variance: {result.sentiment_variance:.4f}")
    print(f"Trajectory: {[f'{s:.2f}' for s in result.sentiment_trajectory]}")

    print(f"\n=== Pattern Densities ===")
    print(f"Hedging: {result.hedging_density:.4f}")
    print(f"Vulnerability: {result.vulnerability_score:.4f}")
    print(f"Confidence variance: {result.confidence_variance:.6f}")

    print(f"\n=== Agent Analysis ===")
    print(f"Sentiment by agent: {result.agent_sentiment}")
    print(f"Hedging by agent: {result.agent_hedging}")

    divergence = compute_agent_affective_divergence(result)
    print(f"\nAgent affective divergence: {divergence:.4f}")
