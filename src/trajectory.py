"""
Trajectory Storage and Geometry Module for MASE.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Ported from Semantic Climate Phase Space with adaptations for
multi-agent dialogue (no biosignal substrate).

This module provides trajectory storage, derivative computation, and
geometric characterization for analyzing dialogue dynamics.

Classes:
    TrajectoryBuffer: Windowed storage of Ψ trajectory with derivatives
    TrajectoryStateVector: First-class trajectory representation per timestep

Functions:
    compute_trajectory_derivatives: Standalone derivative computation
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, asdict
import numpy as np

__all__ = [
    'TrajectoryBuffer',
    'TrajectoryStateVector',
    'compute_trajectory_derivatives',
]


@dataclass
class TrajectoryStateVector:
    """
    First-class trajectory representation computed per timestep.

    Encodes both position and movement through phase space, enabling
    movement-preserving classification that distinguishes states reached
    via different paths.

    Example: "collaborative inquiry from settling" vs "collaborative inquiry from conflict"
    have identical positions but different TSVs due to approach path.

    Attributes:
        position: Current Ψ values {psi_semantic, psi_temporal, psi_affective}
        velocity_magnitude: ||dΨ/dt|| - speed through phase space
        acceleration_magnitude: ||d²Ψ/dt²|| - rate of velocity change
        curvature: Rate of directional change in trajectory
        approach_basin: Basin we transitioned from (if any)
        approach_direction: Unit vector from previous position
        dwell_time: Consecutive turns in current soft region
        movement_annotation: Human-readable movement description
    """
    position: Dict[str, float]
    velocity_magnitude: Optional[float] = None
    acceleration_magnitude: Optional[float] = None
    curvature: Optional[float] = None
    approach_basin: Optional[str] = None
    approach_direction: Optional[Dict[str, float]] = None
    dwell_time: int = 0
    movement_annotation: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class TrajectoryBuffer:
    """
    Windowed storage of Ψ trajectory for derivative and geometry calculation.

    Adapted for MASE multi-agent dialogue:
    - Three substrates: semantic, temporal, affective (no biosignal)
    - Turn-indexed rather than time-indexed

    Stores recent history of Ψ vectors to enable computation of:
    - Velocity (dΨ/dt): First derivative
    - Acceleration (d²Ψ/dt²): Second derivative
    - Curvature: Rate of directional change
    - Tortuosity: Path winding measure
    - Displacement: Start-to-end distance
    - Path length: Total distance traveled
    - Autocorrelation: Memory measure across lags

    Attributes:
        window_size: Maximum number of Ψ observations to retain
        timestep: Time interval between observations (default: 1 turn)
        history: List of (timestamp, psi_vector) tuples
    """

    # Substrates for MASE (no biosignal)
    SUBSTRATES = ['psi_semantic', 'psi_temporal', 'psi_affective']

    def __init__(self, window_size: int = 50, timestep: float = 1.0):
        """
        Initialize trajectory buffer.

        Args:
            window_size: Maximum number of Ψ observations to retain
            timestep: Time interval between observations (default: 1 turn)
        """
        self.window_size = window_size
        self.timestep = timestep
        self.history: List[Tuple[float, dict]] = []

    def append(self, psi_vector: dict, timestamp: float = None) -> None:
        """
        Add new Ψ observation to buffer.

        Args:
            psi_vector: Dict with psi_semantic, psi_temporal, psi_affective
            timestamp: Optional timestamp (if None, use sequential numbering)
        """
        if timestamp is None:
            timestamp = float(len(self.history)) * self.timestep

        self.history.append((timestamp, psi_vector))

        # Maintain window size
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    def clear(self) -> None:
        """Clear all trajectory history."""
        self.history = []

    def __len__(self) -> int:
        """Return number of observations in history."""
        return len(self.history)

    def compute_velocity(self) -> Optional[dict]:
        """
        Calculate dΨ/dt via finite differences.

        Uses central difference: (Ψ[t+1] - Ψ[t-1]) / (2 * dt)

        Returns:
            dict: {
                'dpsi_semantic_dt': float,
                'dpsi_temporal_dt': float,
                'dpsi_affective_dt': float
            } or None if insufficient history
        """
        if len(self.history) < 3:
            return None

        # Central difference: (Ψ[t+1] - Ψ[t-1]) / (2 * dt)
        t_prev, psi_prev = self.history[-3]
        t_next, psi_next = self.history[-1]
        dt = t_next - t_prev

        if dt == 0:
            return None

        velocity = {}
        for key in self.SUBSTRATES:
            prev_val = psi_prev.get(key)
            next_val = psi_next.get(key)

            if prev_val is not None and next_val is not None:
                velocity[f'd{key}_dt'] = (next_val - prev_val) / dt
            else:
                velocity[f'd{key}_dt'] = None

        return velocity

    def compute_acceleration(self) -> Optional[dict]:
        """
        Calculate d²Ψ/dt² via second-order finite differences.

        Uses: (Ψ[t+1] - 2*Ψ[t] + Ψ[t-1]) / dt²

        Returns:
            dict: {
                'd2psi_semantic_dt2': float,
                'd2psi_temporal_dt2': float,
                'd2psi_affective_dt2': float
            } or None if insufficient history
        """
        if len(self.history) < 3:
            return None

        # Second derivative: (Ψ[t+1] - 2*Ψ[t] + Ψ[t-1]) / dt²
        t_prev, psi_prev = self.history[-3]
        t_curr, psi_curr = self.history[-2]
        t_next, psi_next = self.history[-1]

        dt = (t_next - t_prev) / 2  # Average timestep

        if dt == 0:
            return None

        acceleration = {}
        for key in self.SUBSTRATES:
            prev_val = psi_prev.get(key)
            curr_val = psi_curr.get(key)
            next_val = psi_next.get(key)

            if all(v is not None for v in [prev_val, curr_val, next_val]):
                acceleration[f'd2{key}_dt2'] = (next_val - 2*curr_val + prev_val) / (dt ** 2)
            else:
                acceleration[f'd2{key}_dt2'] = None

        return acceleration

    def compute_curvature(self) -> Optional[float]:
        """
        Calculate trajectory curvature (rate of directional change).

        Uses generalized N-D formula: sqrt(||v||² ||a||² - (v·a)²) / ||v||³

        Returns:
            float: Curvature magnitude or None if insufficient history
        """
        velocity = self.compute_velocity()
        acceleration = self.compute_acceleration()

        if velocity is None or acceleration is None:
            return None

        # Extract numeric values only
        v_vals = [v for v in velocity.values() if v is not None]
        a_vals = [v for v in acceleration.values() if v is not None]

        if len(v_vals) < 2 or len(a_vals) < 2:
            return None

        v = np.array(v_vals)
        a = np.array(a_vals[:len(v_vals)])  # Match dimensions

        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            return None

        # Curvature = ||v × a|| / ||v||³ (generalized for N dimensions)
        # In N-D, use: sqrt(||v||² ||a||² - (v·a)²) / ||v||³
        cross_magnitude = np.sqrt(
            max(0, np.linalg.norm(v)**2 * np.linalg.norm(a)**2 - np.dot(v, a)**2)
        )

        return float(cross_magnitude / (v_norm ** 3))

    def compute_tortuosity(self) -> Optional[float]:
        """
        Calculate trajectory tortuosity (path winding measure).

        Tortuosity = path_length / displacement
        - Straight path: tortuosity = 1.0
        - Winding path: tortuosity > 1.0

        Returns:
            float: Tortuosity value >= 1.0, or None if insufficient history
        """
        if len(self.history) < 2:
            return None

        path_length = self.compute_path_length()
        displacement = self.compute_displacement()

        if path_length is None or displacement is None:
            return None

        if displacement < 1e-10:
            # No net displacement - could be circular or stationary
            return None

        return float(path_length / displacement)

    def compute_displacement(self) -> Optional[float]:
        """
        Calculate Euclidean distance from start to end of trajectory.

        Returns:
            float: Displacement distance, or None if insufficient history
        """
        if len(self.history) < 2:
            return None

        _, psi_start = self.history[0]
        _, psi_end = self.history[-1]

        # Extract numeric values
        start_vals = []
        end_vals = []

        for key in self.SUBSTRATES:
            start_v = psi_start.get(key)
            end_v = psi_end.get(key)

            if start_v is not None and end_v is not None:
                start_vals.append(start_v)
                end_vals.append(end_v)

        if not start_vals:
            return None

        start_arr = np.array(start_vals)
        end_arr = np.array(end_vals)

        return float(np.linalg.norm(end_arr - start_arr))

    def compute_path_length(self) -> Optional[float]:
        """
        Calculate total path length (sum of step distances).

        Returns:
            float: Total path length, or None if insufficient history
        """
        if len(self.history) < 2:
            return None

        total_length = 0.0

        for i in range(1, len(self.history)):
            _, psi_prev = self.history[i - 1]
            _, psi_curr = self.history[i]

            # Extract numeric values
            prev_vals = []
            curr_vals = []

            for key in self.SUBSTRATES:
                prev_v = psi_prev.get(key)
                curr_v = psi_curr.get(key)

                if prev_v is not None and curr_v is not None:
                    prev_vals.append(prev_v)
                    curr_vals.append(curr_v)

            if prev_vals:
                prev_arr = np.array(prev_vals)
                curr_arr = np.array(curr_vals)
                total_length += np.linalg.norm(curr_arr - prev_arr)

        return float(total_length)

    def compute_autocorrelation(self, max_lag: int = 10) -> Optional[List[float]]:
        """
        Calculate autocorrelation of Ψ trajectory at multiple lags.

        Autocorrelation measures how similar the trajectory is to itself
        at different time offsets. High autocorrelation indicates memory/persistence.

        Args:
            max_lag: Maximum lag to compute (default: 10)

        Returns:
            List[float]: Autocorrelation values for lags 1 to max_lag,
                        or None if insufficient history
        """
        if len(self.history) < 3:
            return None

        # Extract trajectory as array
        trajectory = []
        for _, psi in self.history:
            vals = []
            for key in self.SUBSTRATES:
                v = psi.get(key)
                if v is not None:
                    vals.append(v)
            if vals:
                trajectory.append(vals)

        if len(trajectory) < 3:
            return None

        trajectory_arr = np.array(trajectory)
        n_points = len(trajectory_arr)

        # Compute composite signal (mean across dimensions)
        composite = np.mean(trajectory_arr, axis=1)

        # Compute autocorrelation for each lag
        autocorr = []
        mean_val = np.mean(composite)
        var_val = np.var(composite)

        if var_val < 1e-10:
            # Constant signal - perfect correlation at all lags
            return [1.0] * min(max_lag, n_points - 1)

        for lag in range(1, min(max_lag + 1, n_points)):
            cov = np.mean((composite[:-lag] - mean_val) * (composite[lag:] - mean_val))
            autocorr.append(float(cov / var_val))

        return autocorr if autocorr else None

    def get_trajectory_segment(self, n_points: int = None) -> Optional[List[Tuple[float, dict]]]:
        """
        Retrieve recent trajectory segment.

        Args:
            n_points: Number of recent points to retrieve (None = all)

        Returns:
            list: List of (timestamp, psi_vector) tuples or None if empty
        """
        if len(self.history) == 0:
            return None

        if n_points is None:
            return self.history.copy()

        return self.history[-n_points:]

    def compute_approach_vector(self) -> Optional[dict]:
        """
        Compute approach vector from previous position to current.

        The approach vector encodes "where you came from" - essential for
        movement-preserving classification. Two positions with identical
        coordinates but different approach vectors represent different states.

        Returns:
            dict: {
                'direction': dict (unit vector per substrate),
                'magnitude': float (step size),
                'speed': float (magnitude / timestep)
            } or None if insufficient history
        """
        if len(self.history) < 2:
            return None

        _, psi_prev = self.history[-2]
        _, psi_curr = self.history[-1]

        # Compute difference vector
        diff = {}
        diff_vals = []
        for key in self.SUBSTRATES:
            prev_val = psi_prev.get(key)
            curr_val = psi_curr.get(key)

            if prev_val is not None and curr_val is not None:
                diff[key] = curr_val - prev_val
                diff_vals.append(diff[key])
            else:
                diff[key] = None

        if not diff_vals:
            return None

        # Compute magnitude
        magnitude = float(np.linalg.norm(diff_vals))

        # Compute unit direction vector
        direction = {}
        if magnitude > 1e-10:
            for key, val in diff.items():
                if val is not None:
                    direction[key] = val / magnitude
                else:
                    direction[key] = None
        else:
            # Stationary - no meaningful direction
            direction = {k: 0.0 if v is not None else None for k, v in diff.items()}

        return {
            'direction': direction,
            'magnitude': magnitude,
            'speed': magnitude / self.timestep
        }

    def compute_velocity_magnitude(self) -> Optional[float]:
        """
        Compute scalar velocity magnitude (speed through phase space).

        Returns:
            float: ||dΨ/dt|| or None if insufficient history
        """
        velocity = self.compute_velocity()
        if velocity is None:
            return None

        v_vals = [v for v in velocity.values() if v is not None]
        if not v_vals:
            return None

        return float(np.linalg.norm(v_vals))

    def compute_acceleration_magnitude(self) -> Optional[float]:
        """
        Compute scalar acceleration magnitude.

        Returns:
            float: ||d²Ψ/dt²|| or None if insufficient history
        """
        acceleration = self.compute_acceleration()
        if acceleration is None:
            return None

        a_vals = [v for v in acceleration.values() if v is not None]
        if not a_vals:
            return None

        return float(np.linalg.norm(a_vals))

    def get_summary(self) -> dict:
        """
        Get summary of current trajectory state.

        Returns:
            dict with velocity, acceleration, curvature, tortuosity, path_length
        """
        return {
            'n_points': len(self.history),
            'velocity_magnitude': self.compute_velocity_magnitude(),
            'acceleration_magnitude': self.compute_acceleration_magnitude(),
            'curvature': self.compute_curvature(),
            'tortuosity': self.compute_tortuosity(),
            'displacement': self.compute_displacement(),
            'path_length': self.compute_path_length(),
        }


def compute_trajectory_derivatives(trajectory: TrajectoryBuffer) -> dict:
    """
    Standalone function for computing trajectory derivatives.

    Convenience function that computes all derivatives from a TrajectoryBuffer.

    Args:
        trajectory: TrajectoryBuffer with recent Ψ history

    Returns:
        dict: {
            'velocity': dict or None,
            'acceleration': dict or None,
            'curvature': float or None,
            'speed': float or None,
            'direction': dict or None
        }
    """
    if trajectory is None or len(trajectory.history) == 0:
        return {
            'velocity': None,
            'acceleration': None,
            'curvature': None,
            'speed': None,
            'direction': None
        }

    velocity = trajectory.compute_velocity()
    acceleration = trajectory.compute_acceleration()
    curvature = trajectory.compute_curvature()

    # Calculate speed and direction from velocity
    speed = None
    direction = None

    if velocity is not None:
        v_vals = [v for v in velocity.values() if v is not None]
        if v_vals:
            speed = float(np.linalg.norm(v_vals))
            if speed > 1e-10:
                direction = {k: v / speed for k, v in velocity.items() if v is not None}

    return {
        'velocity': velocity,
        'acceleration': acceleration,
        'curvature': curvature,
        'speed': speed,
        'direction': direction
    }


# Test if run directly
if __name__ == "__main__":
    print("Trajectory Module Test")
    print("=" * 50)

    # Create trajectory buffer
    buffer = TrajectoryBuffer(window_size=20)

    # Simulate dialogue trajectory (settling into resonance)
    np.random.seed(42)
    for i in range(15):
        # Progressive engagement pattern
        psi = {
            'psi_semantic': 0.1 + i * 0.05 + np.random.randn() * 0.02,
            'psi_temporal': 0.5 + np.random.randn() * 0.05,
            'psi_affective': 0.1 + i * 0.03 + np.random.randn() * 0.01
        }
        buffer.append(psi)

    print(f"\nTrajectory: {len(buffer)} points")

    # Compute derivatives
    summary = buffer.get_summary()
    print(f"\nSummary:")
    for key, val in summary.items():
        if val is not None:
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    # Autocorrelation
    autocorr = buffer.compute_autocorrelation(max_lag=5)
    if autocorr:
        print(f"\nAutocorrelation (lags 1-5): {[f'{a:.3f}' for a in autocorr]}")

    # Approach vector
    approach = buffer.compute_approach_vector()
    if approach:
        print(f"\nApproach vector:")
        print(f"  Direction: {approach['direction']}")
        print(f"  Magnitude: {approach['magnitude']:.4f}")
        print(f"  Speed: {approach['speed']:.4f}")
