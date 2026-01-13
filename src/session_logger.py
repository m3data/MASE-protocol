"""
Session logging for MASE experiments.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Logs dialogue sessions to JSON with metadata for analysis.
Supports incremental checkpointing after each turn.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


@dataclass
class TurnRecord:
    """Record of a single dialogue turn."""
    turn_number: int
    agent_id: str
    agent_name: str
    content: str
    model: str
    temperature: float
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[List[float]] = None  # Stored inline for small sessions


@dataclass
class SessionRecord:
    """Complete record of a dialogue session."""
    session_id: str
    mode: str  # "single_model" or "multi_model"
    provocation_id: Optional[str]
    provocation_text: str
    seed: int
    config_path: Optional[str]
    start_time: str
    end_time: Optional[str] = None
    turns: List[TurnRecord] = field(default_factory=list)

    # Aggregate metadata
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    agent_turn_counts: Dict[str, int] = field(default_factory=dict)

    # Model configuration snapshot
    model_assignments: Dict[str, str] = field(default_factory=dict)
    temperature_assignments: Dict[str, float] = field(default_factory=dict)


class SessionLogger:
    """
    Logger for MASE dialogue sessions.

    Handles JSON serialization with optional numpy array storage
    for embeddings. Supports incremental checkpointing.
    """

    def __init__(
        self,
        output_dir: Path,
        session_id: Optional[str] = None,
        embed_inline: bool = True
    ):
        """
        Initialize session logger.

        Args:
            output_dir: Directory for session output files
            session_id: Optional custom session ID (default: timestamp-based)
            embed_inline: Store embeddings inline in JSON (True) or separate .npy (False)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.embed_inline = embed_inline

        self._session: Optional[SessionRecord] = None
        self._embeddings: List[np.ndarray] = []  # For separate storage

    def start_session(
        self,
        mode: str,
        provocation_text: str,
        seed: int,
        model_assignments: Dict[str, str],
        temperature_assignments: Dict[str, float],
        provocation_id: Optional[str] = None,
        config_path: Optional[str] = None
    ) -> SessionRecord:
        """
        Initialize a new session record.

        Args:
            mode: "single_model" or "multi_model"
            provocation_text: The opening provocation
            seed: Random seed for reproducibility
            model_assignments: Dict mapping agent_id -> model
            temperature_assignments: Dict mapping agent_id -> temperature
            provocation_id: Optional identifier for the provocation
            config_path: Path to config file used

        Returns:
            The initialized SessionRecord
        """
        self._session = SessionRecord(
            session_id=self.session_id,
            mode=mode,
            provocation_id=provocation_id,
            provocation_text=provocation_text,
            seed=seed,
            config_path=config_path,
            start_time=datetime.now().isoformat(),
            model_assignments=model_assignments,
            temperature_assignments=temperature_assignments
        )
        self._embeddings = []

        return self._session

    def log_turn(
        self,
        agent_id: str,
        agent_name: str,
        content: str,
        model: str,
        temperature: float,
        latency_ms: float,
        embedding: Optional[np.ndarray] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        checkpoint: bool = True
    ) -> TurnRecord:
        """
        Log a dialogue turn.

        Args:
            agent_id: Short agent identifier
            agent_name: Full agent name
            content: The agent's response text
            model: Model used for generation
            temperature: Temperature used
            latency_ms: Generation latency in milliseconds
            embedding: Optional embedding vector
            prompt_tokens: Optional prompt token count
            completion_tokens: Optional completion token count
            checkpoint: Save checkpoint after this turn

        Returns:
            The TurnRecord
        """
        if self._session is None:
            raise RuntimeError("Session not started. Call start_session() first.")

        turn_number = len(self._session.turns) + 1

        # Handle embedding storage
        embedding_list = None
        if embedding is not None:
            if self.embed_inline:
                embedding_list = embedding.tolist()
            else:
                self._embeddings.append(embedding)

        turn = TurnRecord(
            turn_number=turn_number,
            agent_id=agent_id,
            agent_name=agent_name,
            content=content,
            model=model,
            temperature=temperature,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            embedding=embedding_list
        )

        self._session.turns.append(turn)

        # Update aggregates
        self._session.total_latency_ms += latency_ms
        if prompt_tokens and completion_tokens:
            self._session.total_tokens += prompt_tokens + completion_tokens

        self._session.agent_turn_counts[agent_id] = \
            self._session.agent_turn_counts.get(agent_id, 0) + 1

        if checkpoint:
            self._save_checkpoint()

        return turn

    def end_session(self) -> Path:
        """
        Finalize and save the session.

        Returns:
            Path to the saved session JSON file
        """
        if self._session is None:
            raise RuntimeError("No active session to end.")

        self._session.end_time = datetime.now().isoformat()

        # Save final files
        json_path = self._save_json()

        if not self.embed_inline and self._embeddings:
            self._save_embeddings()

        return json_path

    def _save_checkpoint(self):
        """Save intermediate checkpoint."""
        self._save_json(suffix="_checkpoint")

    def _save_json(self, suffix: str = "") -> Path:
        """Save session to JSON file."""
        filename = f"session_{self.session_id}{suffix}.json"
        path = self.output_dir / filename

        # Convert to dict for JSON serialization
        data = self._session_to_dict()

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        return path

    def _save_embeddings(self) -> Path:
        """Save embeddings to separate numpy file."""
        filename = f"session_{self.session_id}_embeddings.npy"
        path = self.output_dir / filename

        embeddings_array = np.array(self._embeddings)
        np.save(path, embeddings_array)

        return path

    def _session_to_dict(self) -> Dict[str, Any]:
        """Convert session to JSON-serializable dict."""
        data = {
            "session_id": self._session.session_id,
            "mode": self._session.mode,
            "provocation_id": self._session.provocation_id,
            "provocation_text": self._session.provocation_text,
            "seed": self._session.seed,
            "config_path": self._session.config_path,
            "start_time": self._session.start_time,
            "end_time": self._session.end_time,
            "total_latency_ms": self._session.total_latency_ms,
            "total_tokens": self._session.total_tokens,
            "agent_turn_counts": self._session.agent_turn_counts,
            "model_assignments": self._session.model_assignments,
            "temperature_assignments": self._session.temperature_assignments,
            "turns": [asdict(turn) for turn in self._session.turns]
        }

        # Add embeddings file reference if stored separately
        if not self.embed_inline and self._embeddings:
            data["embeddings_file"] = f"session_{self.session_id}_embeddings.npy"

        return data

    @staticmethod
    def load_session(path: Path) -> Dict[str, Any]:
        """
        Load a saved session from JSON.

        Args:
            path: Path to session JSON file

        Returns:
            Session data as dict
        """
        with open(path) as f:
            data = json.load(f)

        # Load embeddings if stored separately
        if "embeddings_file" in data:
            embeddings_path = path.parent / data["embeddings_file"]
            if embeddings_path.exists():
                data["embeddings"] = np.load(embeddings_path)

        return data


# Test if run directly
if __name__ == "__main__":
    import tempfile

    print("Session Logger Test")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(Path(tmpdir))

        # Start session
        logger.start_session(
            mode="multi_model",
            provocation_text="What is the meaning of success?",
            seed=42,
            model_assignments={"luma": "phi3:latest", "orin": "mistral:latest"},
            temperature_assignments={"luma": 0.6, "orin": 0.5},
            provocation_id="p001"
        )

        # Log some turns
        logger.log_turn(
            agent_id="luma",
            agent_name="luma-child-voice",
            content="What does success even mean? Is it like winning a game?",
            model="phi3:latest",
            temperature=0.6,
            latency_ms=1234.5,
            embedding=np.random.randn(768),
            prompt_tokens=100,
            completion_tokens=20
        )

        logger.log_turn(
            agent_id="orin",
            agent_name="systems-analyst-orin",
            content="Success can be viewed as a system reaching its intended state...",
            model="mistral:latest",
            temperature=0.5,
            latency_ms=2345.6,
            embedding=np.random.randn(768),
            prompt_tokens=150,
            completion_tokens=40
        )

        # End session
        path = logger.end_session()

        print(f"Session saved to: {path}")

        # Load and verify
        data = SessionLogger.load_session(path)
        print(f"Session ID: {data['session_id']}")
        print(f"Mode: {data['mode']}")
        print(f"Turns: {len(data['turns'])}")
        print(f"Total latency: {data['total_latency_ms']:.0f}ms")
        print(f"Agent turn counts: {data['agent_turn_counts']}")
