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

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.cluster import KMeans


# =============================================================================
# Basic Result Dataclasses
# =============================================================================

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


# =============================================================================
# Enhanced Results with Confidence Intervals
# =============================================================================

@dataclass
class CurvatureResultWithCI:
    """Semantic curvature with statistical detail."""
    curvature: float
    curvature_std: float
    confidence_interval: Tuple[float, float]
    p_value: float
    threshold_met: bool
    local_curvatures: List[float] = field(default_factory=list)


@dataclass
class DFAResultWithCI:
    """DFA alpha with fit quality and confidence interval."""
    alpha: float
    r_squared: float
    confidence_interval: Tuple[float, float]
    scales_used: int
    target_range_met: bool  # 0.70 <= alpha <= 0.90


@dataclass
class EntropyResultWithCI:
    """Entropy shift with statistical detail."""
    js_divergence: float
    confidence_interval: Tuple[float, float]
    stability_score: float
    threshold_met: bool
    pre_distribution: List[float] = field(default_factory=list)
    post_distribution: List[float] = field(default_factory=list)
    transition_summary: str = ""


@dataclass
class MetricsResultWithCI:
    """Complete metrics with confidence intervals and significance."""
    # Core metrics
    semantic_curvature: float
    dfa_alpha: float
    entropy_shift: float
    semantic_velocity_mean: float
    semantic_velocity_std: float
    n_turns: int

    # Confidence intervals (95%)
    curvature_ci: Tuple[float, float] = (0.0, 0.0)
    alpha_ci: Tuple[float, float] = (0.5, 0.5)
    entropy_ci: Tuple[float, float] = (0.0, 0.0)

    # Significance
    curvature_p_value: float = 1.0
    alpha_r_squared: float = 0.0
    entropy_stability: float = 0.0

    # Threshold flags (Morgoulis 2025 thresholds)
    curvature_significant: bool = False  # >= 0.35
    alpha_in_range: bool = False         # 0.70-0.90
    entropy_significant: bool = False    # >= 0.12

    # Detailed results (optional)
    curvature_detail: Optional[CurvatureResultWithCI] = None
    dfa_detail: Optional[DFAResultWithCI] = None
    entropy_detail: Optional[EntropyResultWithCI] = None


# Morgoulis (2025) empirically-derived thresholds
THRESHOLDS = {
    'delta_kappa': 0.35,
    'alpha_min': 0.70,
    'alpha_max': 0.90,
    'delta_h': 0.12
}


def _compute_local_curvatures(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute local curvature at each interior point using discrete Frenet-Serret.

    Returns array of local curvatures for bootstrap resampling.
    """
    n = len(embeddings)
    if n < 4:
        return np.array([])

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

    return np.array(local_curvatures)


def semantic_curvature_with_ci(
    embeddings: np.ndarray,
    bootstrap_iterations: int = 500,
    random_state: int = 42
) -> CurvatureResultWithCI:
    """
    Calculate Semantic Curvature (Δκ) with bootstrap confidence interval.

    Args:
        embeddings: Array of shape (n_turns, embedding_dim)
        bootstrap_iterations: Number of bootstrap samples
        random_state: Random seed for reproducibility

    Returns:
        CurvatureResultWithCI with CI and p-value
    """
    np.random.seed(random_state)
    n = len(embeddings)

    if n < 4:
        return CurvatureResultWithCI(
            curvature=0.0,
            curvature_std=0.0,
            confidence_interval=(0.0, 0.0),
            p_value=1.0,
            threshold_met=False,
            local_curvatures=[]
        )

    # Compute local curvatures
    local_curvatures = _compute_local_curvatures(embeddings)

    if len(local_curvatures) == 0:
        return CurvatureResultWithCI(
            curvature=0.0,
            curvature_std=0.0,
            confidence_interval=(0.0, 0.0),
            p_value=1.0,
            threshold_met=False,
            local_curvatures=[]
        )

    curvature = float(np.mean(local_curvatures))
    curvature_std = float(np.std(local_curvatures))

    # Bootstrap confidence interval
    bootstrap_curvatures = []
    for _ in range(bootstrap_iterations):
        boot_indices = np.sort(np.random.choice(n, size=n, replace=True))
        boot_embeddings = embeddings[boot_indices]
        boot_local = _compute_local_curvatures(boot_embeddings)
        if len(boot_local) > 0:
            bootstrap_curvatures.append(np.mean(boot_local))

    if len(bootstrap_curvatures) > 0:
        ci_lower = float(np.percentile(bootstrap_curvatures, 2.5))
        ci_upper = float(np.percentile(bootstrap_curvatures, 97.5))
    else:
        ci_lower, ci_upper = curvature, curvature

    # Statistical significance: compare to shuffled trajectory (null hypothesis)
    null_curvatures = []
    for _ in range(200):
        null_embeddings = np.random.permutation(embeddings)
        null_local = _compute_local_curvatures(null_embeddings)
        if len(null_local) > 0:
            null_curvatures.append(np.mean(null_local))

    if len(null_curvatures) > 0:
        p_value = float(np.mean(np.array(null_curvatures) >= curvature))
    else:
        p_value = 1.0

    return CurvatureResultWithCI(
        curvature=curvature,
        curvature_std=curvature_std,
        confidence_interval=(ci_lower, ci_upper),
        p_value=p_value,
        threshold_met=curvature >= THRESHOLDS['delta_kappa'],
        local_curvatures=local_curvatures.tolist()
    )


def dfa_alpha_with_ci(
    signal: np.ndarray,
    min_scale: int = 4,
    max_scale_fraction: float = 0.25,
    bootstrap_iterations: int = 300,
    random_state: int = 42
) -> DFAResultWithCI:
    """
    Calculate DFA alpha with bootstrap confidence interval and fit quality.

    Args:
        signal: 1D array (semantic velocity over turns)
        min_scale: Minimum window size
        max_scale_fraction: Maximum window as fraction of length
        bootstrap_iterations: Number of bootstrap samples
        random_state: Random seed

    Returns:
        DFAResultWithCI with CI and r_squared
    """
    np.random.seed(random_state)

    def _dfa_with_r2(sig: np.ndarray) -> Tuple[float, float, int]:
        """Internal DFA returning (alpha, r_squared, scales_used)."""
        x = sig - np.mean(sig)
        y = np.cumsum(x)
        N = len(y)

        if N < min_scale * 2:
            return 0.5, 0.0, 0

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
            return 0.5, 0.0, 0

        log_s = np.log10(np.array(valid_scales))
        log_F = np.log10(np.array(F))

        # Remove invalid values
        valid_idx = np.isfinite(log_s) & np.isfinite(log_F)
        log_s = log_s[valid_idx]
        log_F = log_F[valid_idx]

        if len(log_s) < 2:
            return 0.5, 0.0, 0

        alpha, intercept = np.polyfit(log_s, log_F, 1)

        # R-squared
        predicted = alpha * log_s + intercept
        ss_res = np.sum((log_F - predicted)**2)
        ss_tot = np.sum((log_F - np.mean(log_F))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return float(alpha), float(r_squared), len(valid_scales)

    # Primary computation
    alpha, r_squared, scales_used = _dfa_with_r2(signal)

    if len(signal) < 8:
        return DFAResultWithCI(
            alpha=0.5,
            r_squared=0.0,
            confidence_interval=(0.5, 0.5),
            scales_used=0,
            target_range_met=False
        )

    # Bootstrap CI
    bootstrap_alphas = []
    for _ in range(bootstrap_iterations):
        boot_indices = np.random.choice(len(signal), size=len(signal), replace=True)
        boot_signal = signal[boot_indices]
        boot_alpha, _, _ = _dfa_with_r2(boot_signal)
        bootstrap_alphas.append(boot_alpha)

    if len(bootstrap_alphas) > 0:
        ci_lower = float(np.percentile(bootstrap_alphas, 2.5))
        ci_upper = float(np.percentile(bootstrap_alphas, 97.5))
    else:
        ci_lower, ci_upper = alpha, alpha

    return DFAResultWithCI(
        alpha=alpha,
        r_squared=r_squared,
        confidence_interval=(ci_lower, ci_upper),
        scales_used=scales_used,
        target_range_met=THRESHOLDS['alpha_min'] <= alpha <= THRESHOLDS['alpha_max']
    )


def entropy_shift_with_ci(
    embeddings_pre: np.ndarray,
    embeddings_post: np.ndarray,
    n_clusters: int = 8,
    bootstrap_iterations: int = 200,
    random_state: int = 42
) -> EntropyResultWithCI:
    """
    Calculate Entropy Shift (ΔH) with bootstrap CI and transition summary.

    Args:
        embeddings_pre: First half embeddings
        embeddings_post: Second half embeddings
        n_clusters: Number of clusters
        bootstrap_iterations: Number of bootstrap samples
        random_state: Random seed

    Returns:
        EntropyResultWithCI with CI and distributions
    """
    np.random.seed(random_state)

    n_pre = len(embeddings_pre)
    n_post = len(embeddings_post)

    if n_pre < 2 or n_post < 2:
        return EntropyResultWithCI(
            js_divergence=0.0,
            confidence_interval=(0.0, 0.0),
            stability_score=0.0,
            threshold_met=False,
            transition_summary="Insufficient data"
        )

    def _compute_js_with_distributions(
        all_emb: np.ndarray,
        n_pre: int,
        n_k: int
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute JS divergence and return distributions."""
        n_k = min(len(all_emb), n_k)
        if n_k < 2:
            return 0.0, np.array([]), np.array([])

        kmeans = KMeans(n_clusters=n_k, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(all_emb)

        labels_pre = labels[:n_pre]
        labels_post = labels[n_pre:]

        def _dist(lab, n_k):
            counts = np.zeros(n_k)
            for l in lab:
                counts[l] += 1
            return counts / counts.sum()

        p = _dist(labels_pre, n_k)
        q = _dist(labels_post, n_k)

        # JS divergence
        m = 0.5 * (p + q)
        p_safe = np.clip(p, 1e-12, 1)
        q_safe = np.clip(q, 1e-12, 1)
        m_safe = np.clip(m, 1e-12, 1)

        kl_pm = np.sum(p_safe * np.log2(p_safe / m_safe))
        kl_qm = np.sum(q_safe * np.log2(q_safe / m_safe))
        js = 0.5 * kl_pm + 0.5 * kl_qm

        return float(js), p, q

    # Primary computation
    all_embeddings = np.vstack([embeddings_pre, embeddings_post])
    js, pre_dist, post_dist = _compute_js_with_distributions(
        all_embeddings, n_pre, n_clusters
    )

    # Bootstrap CI
    bootstrap_js = []
    for _ in range(bootstrap_iterations):
        boot_pre_idx = np.random.choice(n_pre, n_pre, replace=True)
        boot_post_idx = np.random.choice(n_post, n_post, replace=True)
        boot_all = np.vstack([embeddings_pre[boot_pre_idx], embeddings_post[boot_post_idx]])
        boot_js, _, _ = _compute_js_with_distributions(boot_all, n_pre, n_clusters)
        bootstrap_js.append(boot_js)

    if len(bootstrap_js) > 0:
        ci_lower = float(np.percentile(bootstrap_js, 2.5))
        ci_upper = float(np.percentile(bootstrap_js, 97.5))
        # Stability = 1 - coefficient of variation
        js_std = np.std(bootstrap_js)
        stability = max(0.0, min(1.0, 1 - js_std / (js + 1e-12)))
    else:
        ci_lower, ci_upper = js, js
        stability = 0.0

    # Transition summary
    if len(pre_dist) > 0 and len(post_dist) > 0:
        changes = post_dist - pre_dist
        max_gain_idx = int(np.argmax(changes))
        max_loss_idx = int(np.argmin(changes))

        if js < 0.05:
            summary = "Minimal reorganization - semantic distribution stable"
        elif js < 0.15:
            summary = f"Moderate reorganization - cluster {max_gain_idx} gained (+{changes[max_gain_idx]:.2f}), cluster {max_loss_idx} lost ({changes[max_loss_idx]:.2f})"
        else:
            summary = f"Substantial reorganization (JS={js:.3f}) - significant shift from cluster {max_loss_idx} to {max_gain_idx}"
    else:
        summary = "Could not compute transition"

    return EntropyResultWithCI(
        js_divergence=js,
        confidence_interval=(ci_lower, ci_upper),
        stability_score=stability,
        threshold_met=js >= THRESHOLDS['delta_h'],
        pre_distribution=pre_dist.tolist() if len(pre_dist) > 0 else [],
        post_distribution=post_dist.tolist() if len(post_dist) > 0 else [],
        transition_summary=summary
    )


def compute_metrics_with_ci(
    embeddings: np.ndarray,
    bootstrap_iterations: int = 300,
    random_state: int = 42
) -> MetricsResultWithCI:
    """
    Compute all dialogue metrics with confidence intervals.

    This is the statistically rigorous version for research use.
    For quick analysis, use compute_metrics() instead.

    Args:
        embeddings: Array of shape (n_turns, embedding_dim)
        bootstrap_iterations: Number of bootstrap samples
        random_state: Random seed

    Returns:
        MetricsResultWithCI with all metrics, CIs, and significance
    """
    n = len(embeddings)

    if n < 4:
        return MetricsResultWithCI(
            semantic_curvature=0.0,
            dfa_alpha=0.5,
            entropy_shift=0.0,
            semantic_velocity_mean=0.0,
            semantic_velocity_std=0.0,
            n_turns=n
        )

    # Semantic curvature with CI
    curv_result = semantic_curvature_with_ci(
        embeddings,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state
    )

    # Semantic velocity for DFA
    velocity = semantic_velocity(embeddings)

    # DFA with CI
    dfa_result = dfa_alpha_with_ci(
        velocity,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state
    ) if len(velocity) >= 8 else DFAResultWithCI(
        alpha=0.5,
        r_squared=0.0,
        confidence_interval=(0.5, 0.5),
        scales_used=0,
        target_range_met=False
    )

    # Entropy shift with CI
    mid = n // 2
    entropy_result = entropy_shift_with_ci(
        embeddings[:mid],
        embeddings[mid:],
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state
    ) if mid >= 2 else EntropyResultWithCI(
        js_divergence=0.0,
        confidence_interval=(0.0, 0.0),
        stability_score=0.0,
        threshold_met=False,
        transition_summary="Insufficient data"
    )

    return MetricsResultWithCI(
        # Core metrics
        semantic_curvature=curv_result.curvature,
        dfa_alpha=dfa_result.alpha,
        entropy_shift=entropy_result.js_divergence,
        semantic_velocity_mean=float(np.mean(velocity)) if len(velocity) > 0 else 0.0,
        semantic_velocity_std=float(np.std(velocity)) if len(velocity) > 0 else 0.0,
        n_turns=n,

        # Confidence intervals
        curvature_ci=curv_result.confidence_interval,
        alpha_ci=dfa_result.confidence_interval,
        entropy_ci=entropy_result.confidence_interval,

        # Significance
        curvature_p_value=curv_result.p_value,
        alpha_r_squared=dfa_result.r_squared,
        entropy_stability=entropy_result.stability_score,

        # Threshold flags
        curvature_significant=curv_result.threshold_met,
        alpha_in_range=dfa_result.target_range_met,
        entropy_significant=entropy_result.threshold_met,

        # Detailed results
        curvature_detail=curv_result,
        dfa_detail=dfa_result,
        entropy_detail=entropy_result
    )


# Test if run directly
if __name__ == "__main__":
    print("Metrics Module Test")
    print("=" * 50)

    # Generate synthetic embeddings
    np.random.seed(42)
    n_turns = 30
    dim = 768

    # Random walk (realistic dialogue trajectory)
    embeddings = np.zeros((n_turns, dim))
    embeddings[0] = np.random.randn(dim)
    for i in range(1, n_turns):
        step = np.random.randn(dim) * 0.1
        embeddings[i] = embeddings[i-1] + step
        embeddings[i] /= np.linalg.norm(embeddings[i])  # Normalize

    print(f"\nSynthetic dialogue: {n_turns} turns, {dim} dimensions")

    # Basic metrics (fast)
    print("\n--- Basic Metrics (fast) ---")
    result = compute_metrics(embeddings)

    print(f"  Δκ (curvature):  {result.semantic_curvature:.4f}")
    print(f"  α (DFA):         {result.dfa_alpha:.4f}")
    print(f"  ΔH (entropy):    {result.entropy_shift:.4f}")

    # Enhanced metrics with CI (slower, research-grade)
    print("\n--- Enhanced Metrics with CI ---")
    result_ci = compute_metrics_with_ci(embeddings, bootstrap_iterations=100)

    print(f"  Δκ: {result_ci.semantic_curvature:.4f}  CI: [{result_ci.curvature_ci[0]:.4f}, {result_ci.curvature_ci[1]:.4f}]  p={result_ci.curvature_p_value:.3f}")
    print(f"  α:  {result_ci.dfa_alpha:.4f}  CI: [{result_ci.alpha_ci[0]:.4f}, {result_ci.alpha_ci[1]:.4f}]  R²={result_ci.alpha_r_squared:.3f}")
    print(f"  ΔH: {result_ci.entropy_shift:.4f}  CI: [{result_ci.entropy_ci[0]:.4f}, {result_ci.entropy_ci[1]:.4f}]  stability={result_ci.entropy_stability:.3f}")

    # Threshold flags
    print(f"\n--- Threshold Checks (Morgoulis 2025) ---")
    print(f"  Δκ >= 0.35 (significant): {result_ci.curvature_significant}")
    print(f"  α in [0.70, 0.90]:        {result_ci.alpha_in_range}")
    print(f"  ΔH >= 0.12 (significant): {result_ci.entropy_significant}")

    # Entropy transition summary
    if result_ci.entropy_detail:
        print(f"\n--- Semantic Transition ---")
        print(f"  {result_ci.entropy_detail.transition_summary}")
