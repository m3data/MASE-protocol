"""
Trajectory Integrity Computation Module for MASE.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Ported from Semantic Climate Phase Space with adaptations for
multi-agent dialogue (no biosignal substrate).

This module provides trajectory integrity computation for analyzing
dialogue dynamics. Integrity measures the coherent relationship between
path through phase-space and its own history.

Key concept: Hysteresis - path-dependence, where you are depends on how you arrived.

Score interpretation:
- 0.0-0.3: Fragmented (no memory, random walk)
- 0.3-0.7: Living integrity (responsive persistence)
- 0.7-1.0: Rigid (locked, no responsiveness)

Classes:
    IntegrityAnalyzer: Computes trajectory integrity from path and basin history
    TransformationDetector: Detects constraint-altering events
"""

from typing import Optional, TYPE_CHECKING
import numpy as np

try:
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

if TYPE_CHECKING:
    from .trajectory import TrajectoryBuffer
    from .basins import BasinHistory

__all__ = [
    'IntegrityAnalyzer',
    'TransformationDetector',
    'IntegrityResult',
]


class IntegrityResult:
    """
    Result of integrity analysis.

    Attributes:
        autocorrelation: Memory measure [0, 1]
        tortuosity: Path directness score [0, 1]
        recurrence_rate: Self-intersection [0, 1]
        integrity_score: Composite [0, 1]
        integrity_label: 'fragmented', 'living', 'rigid'
        sufficient_data: Whether trajectory was long enough
    """

    def __init__(
        self,
        autocorrelation: Optional[float] = None,
        tortuosity: Optional[float] = None,
        recurrence_rate: Optional[float] = None,
        integrity_score: float = 0.0,
        integrity_label: str = 'fragmented',
        sufficient_data: bool = False
    ):
        self.autocorrelation = autocorrelation
        self.tortuosity = tortuosity
        self.recurrence_rate = recurrence_rate
        self.integrity_score = integrity_score
        self.integrity_label = integrity_label
        self.sufficient_data = sufficient_data

    def to_dict(self) -> dict:
        return {
            'autocorrelation': self.autocorrelation,
            'tortuosity': self.tortuosity,
            'recurrence_rate': self.recurrence_rate,
            'integrity_score': self.integrity_score,
            'integrity_label': self.integrity_label,
            'sufficient_data': self.sufficient_data
        }


class IntegrityAnalyzer:
    """
    Computes trajectory integrity from path and basin history.

    Integrity = f(autocorrelation, tortuosity, recurrence_rate)

    The composite score weighs three components:
    - Autocorrelation: Memory measure (how past predicts present)
    - Tortuosity: Path directness (winding vs straight)
    - Recurrence: Self-intersection (revisiting previous states)

    Score interpretation:
    - 0.0-0.3: Fragmented (no memory, random walk)
    - 0.3-0.7: Living integrity (responsive persistence)
    - 0.7-1.0: Rigid (locked, no responsiveness)

    In dialogue terms:
    - Fragmented: Conversation jumps randomly, no thread continuity
    - Living: Conversation builds on itself while remaining responsive
    - Rigid: Conversation stuck in repetitive patterns
    """

    # Substrates for MASE (no biosignal)
    SUBSTRATES = ['psi_semantic', 'psi_temporal', 'psi_affective']

    def __init__(
        self,
        autocorr_weight: float = 0.4,
        tortuosity_weight: float = 0.3,
        recurrence_weight: float = 0.3,
        min_trajectory_length: int = 5
    ):
        """
        Initialize integrity analyzer.

        Args:
            autocorr_weight: Weight for autocorrelation component
            tortuosity_weight: Weight for tortuosity component
            recurrence_weight: Weight for recurrence component
            min_trajectory_length: Minimum points needed for valid analysis
        """
        self.weights = {
            'autocorrelation': autocorr_weight,
            'tortuosity': tortuosity_weight,
            'recurrence': recurrence_weight
        }
        self.min_length = min_trajectory_length

    def compute(
        self,
        trajectory: 'TrajectoryBuffer',
        basin_history: 'BasinHistory' = None
    ) -> IntegrityResult:
        """
        Compute trajectory integrity metrics.

        Args:
            trajectory: TrajectoryBuffer with recent Ψ history
            basin_history: Optional BasinHistory for additional context

        Returns:
            IntegrityResult with all computed metrics
        """
        # Handle None or empty trajectory
        if trajectory is None or len(trajectory.history) < self.min_length:
            return IntegrityResult(
                autocorrelation=None,
                tortuosity=None,
                recurrence_rate=None,
                integrity_score=0.0,
                integrity_label='fragmented',
                sufficient_data=False
            )

        # Compute components
        autocorr = self._compute_autocorrelation(trajectory)
        tortuosity = self._compute_tortuosity(trajectory)
        recurrence = self._compute_recurrence_rate(trajectory)

        # Compute composite score
        integrity_score = self._compute_composite(autocorr, tortuosity, recurrence)

        # Label the score
        integrity_label = self._label_integrity(integrity_score)

        return IntegrityResult(
            autocorrelation=autocorr,
            tortuosity=tortuosity,
            recurrence_rate=recurrence,
            integrity_score=integrity_score,
            integrity_label=integrity_label,
            sufficient_data=True
        )

    def _compute_autocorrelation(self, trajectory: 'TrajectoryBuffer') -> float:
        """
        Compute autocorrelation score measuring trajectory memory.

        High autocorrelation = trajectory has memory/persistence
        Low autocorrelation = trajectory is random/memoryless

        Args:
            trajectory: TrajectoryBuffer

        Returns:
            float: Normalized autocorrelation score [0, 1]
        """
        autocorr = trajectory.compute_autocorrelation(max_lag=5)

        if autocorr is None or len(autocorr) == 0:
            return 0.0

        # Use mean of first few lags as measure of memory
        # High positive correlations indicate persistence
        mean_autocorr = np.mean([abs(a) for a in autocorr[:3]])

        # Normalize to [0, 1]
        return float(np.clip(mean_autocorr, 0, 1))

    def _compute_tortuosity(self, trajectory: 'TrajectoryBuffer') -> float:
        """
        Compute tortuosity score (path directness).

        Tortuosity = path_length / displacement
        - Value of 1.0 = perfectly straight
        - Higher values = more winding

        We invert and normalize so:
        - High score = low tortuosity (direct path) -> contributes to rigidity
        - Low score = high tortuosity (winding) -> contributes to living/fragmented

        Args:
            trajectory: TrajectoryBuffer

        Returns:
            float: Normalized tortuosity score [0, 1]
        """
        raw_tortuosity = trajectory.compute_tortuosity()

        if raw_tortuosity is None:
            return 0.5  # Neutral if can't compute

        # Tortuosity >= 1.0 (straight line = 1.0)
        # Transform: 1/tortuosity gives value in (0, 1]
        # Straight line: 1/1 = 1.0 (high directness -> high score -> rigidity)
        # Winding: 1/3 = 0.33 (low directness -> low score -> living/fragmented)
        directness = 1.0 / raw_tortuosity

        return float(np.clip(directness, 0, 1))

    def _compute_recurrence_rate(self, trajectory: 'TrajectoryBuffer') -> float:
        """
        Compute recurrence rate (self-intersection measure).

        Measures how often the trajectory revisits nearby regions.
        - High recurrence = trajectory returns to similar states -> rigidity
        - Low recurrence = trajectory explores new territory -> living/fragmented

        Args:
            trajectory: TrajectoryBuffer

        Returns:
            float: Recurrence rate [0, 1]
        """
        if len(trajectory.history) < 3:
            return 0.0

        # Extract trajectory as array
        traj_points = []
        for _, psi in trajectory.history:
            vals = []
            for key in self.SUBSTRATES:
                v = psi.get(key)
                if v is not None:
                    vals.append(v)
            if vals:
                traj_points.append(vals)

        if len(traj_points) < 3:
            return 0.0

        traj_arr = np.array(traj_points)
        n_points = len(traj_arr)
        threshold = 0.1  # Distance threshold for "nearby"

        if SCIPY_AVAILABLE:
            # Vectorized pairwise distance computation (O(n²) but in C, not Python)
            dist_matrix = squareform(pdist(traj_arr, metric='euclidean'))

            # Get upper triangle excluding diagonal and adjacent pairs (k=2 means skip k diagonals)
            # This gives us pairs (i, j) where j >= i + 2
            upper_mask = np.triu(np.ones((n_points, n_points), dtype=bool), k=2)
            non_adjacent_distances = dist_matrix[upper_mask]

            total_pairs = len(non_adjacent_distances)
            if total_pairs == 0:
                return 0.0

            # Count recurrences (distances below threshold)
            recurrence_count = np.sum(non_adjacent_distances < threshold)
            recurrence_rate = recurrence_count / total_pairs
        else:
            # Fallback: manual computation (slower)
            recurrence_count = 0
            total_pairs = 0
            for i in range(n_points):
                for j in range(i + 2, n_points):
                    dist = np.linalg.norm(traj_arr[i] - traj_arr[j])
                    if dist < threshold:
                        recurrence_count += 1
                    total_pairs += 1

            if total_pairs == 0:
                return 0.0
            recurrence_rate = recurrence_count / total_pairs

        # Scale to make it more sensitive
        return float(np.clip(recurrence_rate * 5, 0, 1))

    def _compute_composite(
        self,
        autocorr: float,
        tortuosity: float,
        recurrence: float
    ) -> float:
        """
        Compute weighted composite integrity score.

        Args:
            autocorr: Autocorrelation score [0, 1]
            tortuosity: Tortuosity score [0, 1]
            recurrence: Recurrence rate [0, 1]

        Returns:
            float: Composite integrity score [0, 1]
        """
        # Handle None values
        autocorr = autocorr if autocorr is not None else 0.5
        tortuosity = tortuosity if tortuosity is not None else 0.5
        recurrence = recurrence if recurrence is not None else 0.0

        composite = (
            self.weights['autocorrelation'] * autocorr +
            self.weights['tortuosity'] * tortuosity +
            self.weights['recurrence'] * recurrence
        )

        return float(np.clip(composite, 0, 1))

    def _label_integrity(self, score: float) -> str:
        """
        Label integrity score.

        Args:
            score: Integrity score [0, 1]

        Returns:
            str: 'fragmented', 'living', or 'rigid'
        """
        if score < 0.3:
            return 'fragmented'
        elif score > 0.7:
            return 'rigid'
        else:
            return 'living'


class TransformationDetector:
    """
    Detects constraint-altering events in trajectory.

    Future work - enables transformation-indexed (vs turn-indexed) analysis.
    Currently stubbed with basic implementation.

    A transformation is defined as:
    - Large change in any substrate (> threshold)
    - Basin transition (if basin_transition_counts=True)

    In dialogue terms, a transformation might be:
    - A topic shift that changes the semantic trajectory
    - An emotional shift that alters affective dynamics
    - A breakthrough moment that reorients the conversation
    """

    # Substrates for MASE (no biosignal)
    SUBSTRATES = ['psi_semantic', 'psi_temporal', 'psi_affective']

    def __init__(
        self,
        substrate_threshold: float = 0.3,
        basin_transition_counts: bool = True
    ):
        """
        Initialize transformation detector.

        Args:
            substrate_threshold: Minimum substrate change to count as transformation
            basin_transition_counts: Whether basin transitions count as transformations
        """
        self.substrate_threshold = substrate_threshold
        self.basin_transition_counts = basin_transition_counts

    def detect(
        self,
        psi_prev: dict,
        psi_curr: dict,
        basin_prev: str = None,
        basin_curr: str = None
    ) -> bool:
        """
        Detect if this step represents a constraint-altering event.

        Args:
            psi_prev: Previous Ψ vector
            psi_curr: Current Ψ vector
            basin_prev: Previous basin (optional)
            basin_curr: Current basin (optional)

        Returns:
            bool: True if this is a transformation
        """
        # Check basin transition
        if self.basin_transition_counts:
            if basin_prev is not None and basin_curr is not None:
                if basin_prev != basin_curr:
                    return True

        # Check substrate changes
        for key in self.SUBSTRATES:
            prev_val = psi_prev.get(key)
            curr_val = psi_curr.get(key)

            if prev_val is not None and curr_val is not None:
                if abs(curr_val - prev_val) > self.substrate_threshold:
                    return True

        return False

    def compute_transformation_density(
        self,
        trajectory: 'TrajectoryBuffer',
        basin_history: 'BasinHistory' = None
    ) -> float:
        """
        Compute transformations per turn.

        Args:
            trajectory: TrajectoryBuffer with Ψ history
            basin_history: BasinHistory with basin sequence

        Returns:
            float: Transformation density [0, 1]
        """
        if trajectory is None or len(trajectory.history) < 2:
            return 0.0

        transformations = 0
        basin_sequence = basin_history.get_basin_sequence() if basin_history else [None] * len(trajectory.history)

        for i in range(1, len(trajectory.history)):
            _, psi_prev = trajectory.history[i - 1]
            _, psi_curr = trajectory.history[i]

            basin_prev = basin_sequence[i - 1] if i - 1 < len(basin_sequence) else None
            basin_curr = basin_sequence[i] if i < len(basin_sequence) else None

            if self.detect(psi_prev, psi_curr, basin_prev, basin_curr):
                transformations += 1

        # Normalize by number of steps
        n_steps = len(trajectory.history) - 1
        density = transformations / n_steps if n_steps > 0 else 0.0

        return float(np.clip(density, 0, 1))


# Test if run directly
if __name__ == "__main__":
    print("Integrity Module Test")
    print("=" * 50)

    # Import trajectory locally for test
    from trajectory import TrajectoryBuffer
    from basins import BasinHistory

    # Create trajectory buffer
    buffer = TrajectoryBuffer(window_size=20)
    history = BasinHistory()

    # Simulate different dialogue patterns

    print("\n--- Test 1: Fragmented Dialogue (random jumps) ---")
    buffer.clear()
    np.random.seed(42)
    for i in range(10):
        psi = {
            'psi_semantic': np.random.uniform(-1, 1),
            'psi_temporal': np.random.uniform(0, 1),
            'psi_affective': np.random.uniform(-1, 1)
        }
        buffer.append(psi)

    analyzer = IntegrityAnalyzer()
    result = analyzer.compute(buffer)
    print(f"  Integrity: {result.integrity_score:.3f} ({result.integrity_label})")
    print(f"  Autocorr: {result.autocorrelation:.3f}, Tortuosity: {result.tortuosity:.3f}, Recurrence: {result.recurrence_rate:.3f}")

    print("\n--- Test 2: Living Dialogue (progressive engagement) ---")
    buffer.clear()
    np.random.seed(42)
    for i in range(10):
        # Progressive but responsive pattern
        psi = {
            'psi_semantic': 0.1 + i * 0.08 + np.random.randn() * 0.05,
            'psi_temporal': 0.5 + np.random.randn() * 0.1,
            'psi_affective': 0.1 + i * 0.06 + np.random.randn() * 0.03
        }
        buffer.append(psi)

    result = analyzer.compute(buffer)
    print(f"  Integrity: {result.integrity_score:.3f} ({result.integrity_label})")
    print(f"  Autocorr: {result.autocorrelation:.3f}, Tortuosity: {result.tortuosity:.3f}, Recurrence: {result.recurrence_rate:.3f}")

    print("\n--- Test 3: Rigid Dialogue (oscillating pattern) ---")
    buffer.clear()
    np.random.seed(42)
    for i in range(10):
        # Oscillating between two states
        phase = i % 2
        psi = {
            'psi_semantic': 0.3 if phase == 0 else 0.35,
            'psi_temporal': 0.5 + np.random.randn() * 0.02,
            'psi_affective': 0.2 if phase == 0 else 0.25
        }
        buffer.append(psi)

    result = analyzer.compute(buffer)
    print(f"  Integrity: {result.integrity_score:.3f} ({result.integrity_label})")
    print(f"  Autocorr: {result.autocorrelation:.3f}, Tortuosity: {result.tortuosity:.3f}, Recurrence: {result.recurrence_rate:.3f}")

    print("\n--- Transformation Detection Test ---")
    buffer.clear()
    np.random.seed(42)

    # Build trajectory with one big shift
    for i in range(5):
        psi = {
            'psi_semantic': 0.2 + np.random.randn() * 0.02,
            'psi_temporal': 0.5,
            'psi_affective': 0.1
        }
        buffer.append(psi)

    # Big transformation
    psi = {
        'psi_semantic': 0.8,  # Jump!
        'psi_temporal': 0.5,
        'psi_affective': 0.5  # Jump!
    }
    buffer.append(psi)

    for i in range(4):
        psi = {
            'psi_semantic': 0.75 + np.random.randn() * 0.02,
            'psi_temporal': 0.5,
            'psi_affective': 0.45 + np.random.randn() * 0.02
        }
        buffer.append(psi)

    detector = TransformationDetector(substrate_threshold=0.3)
    density = detector.compute_transformation_density(buffer)
    print(f"  Transformation density: {density:.3f}")
