"""
Semantic metrics for MASE dialogue analysis.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Core metrics ported from Semantic Climate Phase Space, based on
Morgoulis (2025): https://github.com/daryamorgoulis/4d-semantic-coupling

Metrics:
- Semantic Curvature (Δκ): Local trajectory bending via Frenet-Serret
- Fractal Similarity (α): DFA scaling exponent for self-organization
- Entropy Shift (ΔH): Jensen-Shannon divergence for semantic reorganization
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class MetricsResult:
    """Results from computing dialogue metrics."""
    semantic_curvature: float           # Δκ - trajectory bending
    dfa_alpha: float                    # α - fractal similarity
    entropy_shift: float                # ΔH - semantic reorganization
    semantic_velocity_mean: float       # Mean inter-turn distance
    semantic_velocity_std: float        # Velocity variability
    n_turns: int                        # Number of turns analyzed


def semantic_curvature(embeddings: np.ndarray) -> float:
    """
    Calculate Semantic Curvature (Δκ) for a dialogue trajectory.

    Uses discrete local curvature via Frenet-Serret formula.
    Measures how fast the direction of semantic movement changes.

    Args:
        embeddings: Array of shape (n_turns, embedding_dim)

    Returns:
        float: Mean local curvature κ = ||a_perp|| / ||v||²

    Reference:
        Morgoulis (2025), fixed implementation per 2025-12-08 review
    """
    n = embeddings.shape[0]
    if n < 4:
        return 0.0

    velocities = np.diff(embeddings, axis=0)
    accelerations = np.diff(velocities, axis=0)

    local_curvatures = []
    for i in range(len(accelerations)):
        v = velocities[i]
        a = accelerations[i]

        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            local_curvatures.append(0.0)
            continue

        v_hat = v / v_norm
        a_parallel = np.dot(a, v_hat) * v_hat
        a_perp = a - a_parallel
        kappa = np.linalg.norm(a_perp) / (v_norm ** 2)
        local_curvatures.append(kappa)

    return float(np.mean(local_curvatures)) if local_curvatures else 0.0


def dfa_alpha(
    signal: np.ndarray,
    min_scale: int = 4,
    max_scale_fraction: float = 0.25
) -> float:
    """
    Calculate Fractal Similarity Score (α) via Detrended Fluctuation Analysis.

    Interpretation:
    - α ≈ 0.5: White noise (uncorrelated)
    - α ≈ 1.0: 1/f noise (pink noise, healthy complexity)
    - α ≈ 1.5: Brownian motion (over-correlated)
    - Target range [0.70, 0.90]: Self-organization without rigidity

    Args:
        signal: 1D array (semantic velocity over turns)
        min_scale: Minimum window size
        max_scale_fraction: Maximum window as fraction of length

    Returns:
        float: DFA scaling exponent α

    Reference:
        Morgoulis (2025), based on Peng et al. (1994)
    """
    x = signal - np.mean(signal)
    y = np.cumsum(x)
    N = len(y)

    if N < min_scale * 2:
        return 0.5

    max_scale = max(min(int(N * max_scale_fraction), N // 2), min_scale + 1)
    scales = np.unique(np.logspace(
        np.log10(min_scale),
        np.log10(max_scale),
        16
    ).astype(int))

    F = []
    valid_scales = []

    for s in scales:
        nseg = N // s
        if nseg < 2:
            continue

        segs = y[:nseg * s].reshape(nseg, s)
        rms = []

        for seg in segs:
            t_idx = np.arange(s)
            coeff = np.polyfit(t_idx, seg, 1)
            trend = np.polyval(coeff, t_idx)
            detr = seg - trend
            rms.append(np.sqrt(np.mean(detr**2)))

        F.append(np.mean(rms))
        valid_scales.append(s)

    if len(F) < 2:
        return 0.5

    log_s = np.log10(np.array(valid_scales))
    log_F = np.log10(np.array(F))
    alpha, _ = np.polyfit(log_s, log_F, 1)

    return float(alpha)


def entropy_shift(
    embeddings_pre: np.ndarray,
    embeddings_post: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 42
) -> float:
    """
    Calculate Entropy Shift (ΔH) between dialogue halves.

    Uses shared clustering + Jensen-Shannon divergence to measure
    semantic reorganization. High values indicate the conversation
    moved to different regions of semantic space.

    Args:
        embeddings_pre: First half embeddings, shape (n, d)
        embeddings_post: Second half embeddings, shape (m, d)
        n_clusters: Number of clusters for KMeans
        random_state: Random seed

    Returns:
        float: Jensen-Shannon divergence [0, 1]

    Reference:
        Morgoulis (2025), fixed with shared clustering per 2025-12-08
    """
    n_pre = len(embeddings_pre)
    n_post = len(embeddings_post)

    if n_pre < 2 or n_post < 2:
        return 0.0

    n_clusters = min(n_pre + n_post, n_clusters)
    if n_clusters < 2:
        return 0.0

    all_embeddings = np.vstack([embeddings_pre, embeddings_post])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(all_embeddings)

    labels_pre = labels[:n_pre]
    labels_post = labels[n_pre:]

    def _distribution(labels_subset, n_k):
        counts = np.zeros(n_k)
        for label in labels_subset:
            counts[label] += 1
        return counts / counts.sum()

    p = _distribution(labels_pre, n_clusters)
    q = _distribution(labels_post, n_clusters)

    m = 0.5 * (p + q)

    def _kl_divergence(pk, qk):
        pk = np.clip(pk, 1e-12, 1)
        qk = np.clip(qk, 1e-12, 1)
        return float(np.sum(pk * np.log2(pk / qk)))

    js = 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
    return float(js)


def semantic_velocity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute semantic velocity (consecutive cosine distances).

    Args:
        embeddings: Array of shape (n_turns, embedding_dim)

    Returns:
        Array of velocities, shape (n_turns - 1,)
    """
    if len(embeddings) < 2:
        return np.array([])

    velocities = []
    for i in range(1, len(embeddings)):
        e1, e2 = embeddings[i-1], embeddings[i]
        n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if n1 == 0 or n2 == 0:
            velocities.append(1.0)
        else:
            sim = np.dot(e1, e2) / (n1 * n2)
            velocities.append(1.0 - sim)

    return np.array(velocities)


def compute_metrics(embeddings: np.ndarray, seed: int = 42) -> MetricsResult:
    """
    Compute all dialogue metrics from embeddings.

    Args:
        embeddings: Array of shape (n_turns, embedding_dim)
        seed: Random seed for entropy_shift clustering

    Returns:
        MetricsResult with all computed metrics
    """
    n = len(embeddings)

    # Semantic curvature
    kappa = semantic_curvature(embeddings)

    # Semantic velocity for DFA
    velocity = semantic_velocity(embeddings)
    alpha = dfa_alpha(velocity) if len(velocity) >= 8 else 0.5

    # Entropy shift (first half vs second half)
    mid = n // 2
    delta_h = entropy_shift(
        embeddings[:mid],
        embeddings[mid:],
        random_state=seed
    ) if mid >= 2 else 0.0

    return MetricsResult(
        semantic_curvature=kappa,
        dfa_alpha=alpha,
        entropy_shift=delta_h,
        semantic_velocity_mean=float(np.mean(velocity)) if len(velocity) > 0 else 0.0,
        semantic_velocity_std=float(np.std(velocity)) if len(velocity) > 0 else 0.0,
        n_turns=n
    )


def compute_metrics_from_session(session_data: Dict[str, Any]) -> MetricsResult:
    """
    Compute metrics from a loaded session JSON.

    Args:
        session_data: Session dict from SessionLogger.load_session()

    Returns:
        MetricsResult
    """
    turns = session_data.get("turns", [])

    embeddings = []
    for turn in turns:
        emb = turn.get("embedding")
        if emb is not None:
            embeddings.append(np.array(emb))

    if len(embeddings) < 4:
        return MetricsResult(
            semantic_curvature=0.0,
            dfa_alpha=0.5,
            entropy_shift=0.0,
            semantic_velocity_mean=0.0,
            semantic_velocity_std=0.0,
            n_turns=len(embeddings)
        )

    embeddings = np.array(embeddings)
    seed = session_data.get("seed", 42)

    return compute_metrics(embeddings, seed=seed)


# Test if run directly
if __name__ == "__main__":
    print("Metrics Module Test")
    print("=" * 50)

    # Generate synthetic embeddings
    np.random.seed(42)
    n_turns = 20
    dim = 768

    # Random walk (realistic dialogue trajectory)
    embeddings = np.zeros((n_turns, dim))
    embeddings[0] = np.random.randn(dim)
    for i in range(1, n_turns):
        step = np.random.randn(dim) * 0.1
        embeddings[i] = embeddings[i-1] + step
        embeddings[i] /= np.linalg.norm(embeddings[i])  # Normalize

    print(f"\nSynthetic dialogue: {n_turns} turns, {dim} dimensions")

    # Compute metrics
    result = compute_metrics(embeddings)

    print(f"\nMetrics:")
    print(f"  Δκ (curvature):  {result.semantic_curvature:.4f}")
    print(f"  α (DFA):         {result.dfa_alpha:.4f}")
    print(f"  ΔH (entropy):    {result.entropy_shift:.4f}")
    print(f"  Velocity mean:   {result.semantic_velocity_mean:.4f}")
    print(f"  Velocity std:    {result.semantic_velocity_std:.4f}")

    # Interpretation
    print(f"\nInterpretation:")
    if 0.70 <= result.dfa_alpha <= 0.90:
        print(f"  α in healthy range [0.70-0.90]: self-organizing complexity")
    elif result.dfa_alpha < 0.70:
        print(f"  α < 0.70: tendency toward noise/randomness")
    else:
        print(f"  α > 0.90: tendency toward rigidity/repetition")
