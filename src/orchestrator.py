"""
Dialogue orchestration for MASE multi-agent conversations.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Coordinates turn-taking between agents with deterministic
selection logic for experimental reproducibility.
"""

import re
import json
import random
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

from .ollama_client import OllamaClient, ModelWarmthManager
from .agents import Agent, EnsembleConfig, load_ensemble
from .embedding_service import EmbeddingService
from .session_logger import SessionLogger, TurnRecord


@dataclass
class TurnError:
    """Record of a failed turn attempt."""
    turn_number: int
    agent_id: str
    model: str
    error_type: str
    error_message: str
    attempt: int
    timestamp: str


@dataclass
class DialogueContext:
    """Context passed to agents for generation."""
    provocation: str
    recent_turns: List[Tuple[str, str]]  # List of (agent_name, content)
    all_agent_names: List[str]


class TurnSelector:
    """
    Selects which agent speaks next.

    Uses deterministic logic with seeded randomness for reproducibility:
    1. Never same speaker twice in a row
    2. If an agent is mentioned by name, they respond
    3. Otherwise, weighted random favoring underrepresented voices
    """

    def __init__(self, agents: Dict[str, Agent], seed: int):
        """
        Initialize turn selector.

        Args:
            agents: Dict of available agents
            seed: Random seed for reproducibility
        """
        self.agents = agents
        self.agent_ids = list(agents.keys())
        self.rng = random.Random(seed)

        # Track turn counts for balancing
        self.turn_counts: Dict[str, int] = {aid: 0 for aid in self.agent_ids}
        self.last_speaker: Optional[str] = None

    def select_next(
        self,
        last_content: Optional[str] = None,
        force_agent: Optional[str] = None
    ) -> str:
        """
        Select the next speaker.

        Args:
            last_content: Content of the last turn (for mention detection)
            force_agent: Force a specific agent (e.g., opening agent)

        Returns:
            Agent ID of next speaker
        """
        if force_agent and force_agent in self.agents:
            return self._select(force_agent)

        # Get eligible agents (exclude last speaker)
        eligible = [aid for aid in self.agent_ids if aid != self.last_speaker]

        if not eligible:
            eligible = self.agent_ids  # Fallback if only one agent

        # Check for mentions in last content
        if last_content:
            mentioned = self._detect_mentions(last_content)
            mentioned_eligible = [aid for aid in mentioned if aid in eligible]
            if mentioned_eligible:
                # Pick first mentioned agent that's eligible
                return self._select(mentioned_eligible[0])

        # Weighted random selection favoring underrepresented voices
        return self._select(self._weighted_choice(eligible))

    def _detect_mentions(self, content: str) -> List[str]:
        """Detect agent mentions in content."""
        content_lower = content.lower()
        mentioned = []

        for agent_id, agent in self.agents.items():
            # Check for agent ID or name mention
            if agent_id.lower() in content_lower:
                mentioned.append(agent_id)
            elif agent.name and agent.name.split('-')[0].lower() in content_lower:
                mentioned.append(agent_id)

        return mentioned

    def _weighted_choice(self, eligible: List[str]) -> str:
        """Choose agent with weights inversely proportional to turn count."""
        if not eligible:
            return self.rng.choice(self.agent_ids)

        # Calculate weights (inverse of turn count + 1 to avoid division by zero)
        max_turns = max(self.turn_counts.values()) + 1
        weights = []
        for aid in eligible:
            # Higher weight for agents who have spoken less
            weight = max_turns - self.turn_counts[aid] + 1
            weights.append(weight)

        # Normalize and select
        total = sum(weights)
        r = self.rng.random() * total
        cumulative = 0
        for aid, weight in zip(eligible, weights):
            cumulative += weight
            if r <= cumulative:
                return aid

        return eligible[-1]  # Fallback

    def _select(self, agent_id: str) -> str:
        """Record selection and return agent ID."""
        self.turn_counts[agent_id] += 1
        self.last_speaker = agent_id
        return agent_id


class DialogueOrchestrator:
    """
    Orchestrates multi-agent dialogue sessions.

    Coordinates:
    - Agent loading and model assignment
    - Turn-by-turn generation via Ollama
    - Embedding generation for semantic analysis
    - Session logging with checkpoints
    - Turn-level error recovery
    - Model warmth management for long runs
    """

    def __init__(
        self,
        config: EnsembleConfig,
        agents_dir: Optional[Path] = None,
        ollama_base_url: str = "http://localhost:11434",
        turn_retries: int = 3,
        turn_retry_backoff: float = 2.0,
        keep_models_warm: bool = True
    ):
        """
        Initialize orchestrator.

        Args:
            config: Ensemble configuration
            agents_dir: Path to agent definitions
            ollama_base_url: Ollama server URL
            turn_retries: Max retries per turn on failure
            turn_retry_backoff: Exponential backoff base for retries
            keep_models_warm: Whether to keep models loaded during long runs
        """
        self.config = config
        self.agents = load_ensemble(agents_dir, config)
        self.ollama = OllamaClient(base_url=ollama_base_url)
        self.embedding_service: Optional[EmbeddingService] = None

        # Error recovery settings
        self.turn_retries = turn_retries
        self.turn_retry_backoff = turn_retry_backoff
        self.keep_models_warm = keep_models_warm

        # Runtime state
        self._warmth_manager: Optional[ModelWarmthManager] = None
        self._turn_errors: List[TurnError] = []
        self._start_time: Optional[datetime] = None

    def run_dialogue(
        self,
        provocation: str,
        output_dir: Path,
        max_turns: Optional[int] = None,
        seed: int = 42,
        provocation_id: Optional[str] = None,
        config_path: Optional[str] = None,
        compute_embeddings: bool = True,
        opening_agent: Optional[str] = None,
        context_window: Optional[int] = None,
        resume_from: Optional[Path] = None
    ) -> Path:
        """
        Run a complete dialogue session.

        Args:
            provocation: The opening provocation text
            output_dir: Directory for session output
            max_turns: Maximum turns (default from config)
            seed: Random seed for reproducibility
            provocation_id: Optional provocation identifier
            config_path: Path to config file (for logging)
            compute_embeddings: Whether to compute embeddings
            opening_agent: First agent to speak (default from config)
            context_window: Number of recent turns in context (default from config)
            resume_from: Path to checkpoint file to resume from

        Returns:
            Path to saved session JSON
        """
        # Resolve defaults from config
        max_turns = max_turns or self.config.dialogue_max_turns
        opening_agent = opening_agent or self.config.dialogue_opening_agent
        context_window = context_window or self.config.dialogue_context_window

        # Initialize components
        if compute_embeddings and self.embedding_service is None:
            self.embedding_service = EmbeddingService()

        turn_selector = TurnSelector(self.agents, seed)
        logger = SessionLogger(output_dir)

        # Build model/temperature assignment dicts
        model_assignments = {aid: self.config.get_model_for_agent(aid)
                           for aid in self.agents}
        temp_assignments = {aid: self.config.get_temperature_for_agent(aid)
                          for aid in self.agents}

        # Handle resume from checkpoint
        start_turn = 1
        dialogue_history: List[Tuple[str, str, str]] = []

        if resume_from and resume_from.exists():
            checkpoint_data = self._load_checkpoint(resume_from)
            if checkpoint_data:
                dialogue_history = checkpoint_data["history"]
                start_turn = checkpoint_data["next_turn"]
                # Reconstruct turn selector state
                for agent_id, _, _ in dialogue_history:
                    turn_selector._select(agent_id)
                print(f"\n  [Resume] Loaded checkpoint with {len(dialogue_history)} turns, starting at turn {start_turn}")

        # Start session (fresh or resumed)
        logger.start_session(
            mode=self.config.mode,
            provocation_text=provocation,
            seed=seed,
            model_assignments=model_assignments,
            temperature_assignments=temp_assignments,
            provocation_id=provocation_id,
            config_path=config_path
        )

        # Re-log existing turns from checkpoint
        for agent_id, agent_name, content in dialogue_history:
            logger.log_turn(
                agent_id=agent_id,
                agent_name=agent_name,
                content=content,
                model=model_assignments[agent_id],
                temperature=temp_assignments[agent_id],
                latency_ms=0,  # Unknown from checkpoint
                checkpoint=False  # Don't re-checkpoint
            )

        # Start model warmth management
        if self.keep_models_warm:
            models = list(set(model_assignments.values()))
            self._warmth_manager = ModelWarmthManager(self.ollama, models)
            self._warmth_manager.start()

        self._start_time = datetime.now()
        self._turn_errors = []

        print(f"\n{'='*60}")
        print(f"MASE Dialogue Session")
        print(f"Mode: {self.config.mode}")
        print(f"Seed: {seed}")
        print(f"Turns: {start_turn}-{max_turns} ({max_turns - start_turn + 1} remaining)")
        print(f"Models: {', '.join(set(model_assignments.values()))}")
        print(f"{'='*60}\n")
        print(f"Provocation:\n{provocation}\n")
        print(f"{'='*60}\n")

        try:
            # Main dialogue loop
            for turn_num in range(start_turn, max_turns + 1):
                # Progress indicator
                self._print_progress(turn_num, max_turns)

                # Select next speaker
                last_content = dialogue_history[-1][2] if dialogue_history else None
                force = opening_agent if turn_num == 1 else None
                agent_id = turn_selector.select_next(last_content, force)
                agent = self.agents[agent_id]

                # Build context for this turn
                context = self._build_context(
                    agent=agent,
                    provocation=provocation,
                    dialogue_history=dialogue_history,
                    context_window=context_window
                )

                # Generate response with retry logic
                print(f"[Turn {turn_num}/{max_turns}] {agent_id} ({agent.model})...")

                response_text, metadata = self._generate_with_retry(
                    agent=agent,
                    context=context,
                    seed=seed + turn_num,
                    turn_num=turn_num
                )

                # Mark model as recently used
                if self._warmth_manager:
                    self._warmth_manager.touch(agent.model)

                print(f"  Latency: {metadata.latency_ms:.0f}ms | Tokens: {metadata.total_tokens or '?'}")
                print(f"  {agent_id}: {response_text[:100]}{'...' if len(response_text) > 100 else ''}\n")

                # Compute embedding
                embedding = None
                if compute_embeddings and self.embedding_service:
                    embedding = self.embedding_service.embed(response_text)

                # Log turn (with checkpoint)
                logger.log_turn(
                    agent_id=agent_id,
                    agent_name=agent.name,
                    content=response_text,
                    model=agent.model,
                    temperature=agent.temperature,
                    latency_ms=metadata.latency_ms,
                    embedding=embedding,
                    prompt_tokens=metadata.prompt_tokens,
                    completion_tokens=metadata.completion_tokens
                )

                # Update history
                dialogue_history.append((agent_id, agent.name, response_text))

        finally:
            # Always clean up warmth manager
            if self._warmth_manager:
                self._warmth_manager.stop()
                self._warmth_manager = None

        # End session
        session_path = logger.end_session()

        # Print summary
        elapsed = datetime.now() - self._start_time
        print(f"{'='*60}")
        print(f"Session complete in {self._format_duration(elapsed)}")
        print(f"Turns: {len(dialogue_history)} | Errors recovered: {len(self._turn_errors)}")
        print(f"Saved to: {session_path}")
        print(f"{'='*60}\n")

        return session_path

    def _generate_with_retry(
        self,
        agent: Agent,
        context: List[Dict[str, str]],
        seed: int,
        turn_num: int
    ) -> Tuple[str, 'ResponseMetadata']:
        """
        Generate a response with retry logic on failure.

        Args:
            agent: Agent to generate for
            context: Message context
            seed: Random seed
            turn_num: Current turn number

        Returns:
            Tuple of (response_text, metadata)

        Raises:
            RuntimeError: If all retries exhausted
        """
        from .ollama_client import ResponseMetadata

        last_error = None
        for attempt in range(self.turn_retries):
            try:
                return self.ollama.generate(
                    model=agent.model,
                    messages=context,
                    temperature=agent.temperature,
                    seed=seed
                )
            except (TimeoutError, ConnectionError, RuntimeError) as e:
                last_error = e
                error_record = TurnError(
                    turn_number=turn_num,
                    agent_id=agent.id,
                    model=agent.model,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    attempt=attempt + 1,
                    timestamp=datetime.now().isoformat()
                )
                self._turn_errors.append(error_record)

                if attempt < self.turn_retries - 1:
                    wait = self.turn_retry_backoff ** attempt
                    print(f"  [Retry] {type(e).__name__} on attempt {attempt + 1}, waiting {wait:.0f}s...")
                    time.sleep(wait)

        raise RuntimeError(
            f"Turn {turn_num} failed after {self.turn_retries} attempts: {last_error}"
        )

    def _load_checkpoint(self, path: Path) -> Optional[Dict]:
        """Load dialogue state from checkpoint file."""
        try:
            with open(path) as f:
                data = json.load(f)

            history = [
                (t["agent_id"], t["agent_name"], t["content"])
                for t in data.get("turns", [])
            ]

            return {
                "history": history,
                "next_turn": len(history) + 1
            }
        except Exception as e:
            print(f"  [Resume] Failed to load checkpoint: {e}")
            return None

    def _print_progress(self, current: int, total: int):
        """Print progress indicator with ETA."""
        if self._start_time is None:
            return

        elapsed = datetime.now() - self._start_time
        if current > 1:
            avg_per_turn = elapsed.total_seconds() / (current - 1)
            remaining = (total - current + 1) * avg_per_turn
            eta = timedelta(seconds=int(remaining))
            print(f"  [Progress] {current}/{total} | Elapsed: {self._format_duration(elapsed)} | ETA: {self._format_duration(eta)}")

    @staticmethod
    def _format_duration(td: timedelta) -> str:
        """Format timedelta as human-readable string."""
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m"
        elif minutes > 0:
            return f"{minutes}m{seconds:02d}s"
        else:
            return f"{seconds}s"

    def _build_context(
        self,
        agent: Agent,
        provocation: str,
        dialogue_history: List[Tuple[str, str, str]],
        context_window: int
    ) -> List[Dict[str, str]]:
        """
        Build the message context for an agent.

        Args:
            agent: The agent who will respond
            provocation: The opening provocation
            dialogue_history: Full dialogue history
            context_window: Number of recent turns to include

        Returns:
            List of messages in chat format
        """
        messages = []

        # System prompt (agent persona)
        messages.append({
            "role": "system",
            "content": self._build_system_prompt(agent, provocation)
        })

        # Recent dialogue as context
        recent = dialogue_history[-context_window:] if dialogue_history else []

        for agent_id, agent_name, content in recent:
            # Format as user messages (the agent sees others' contributions)
            speaker_label = agent_name.split('-')[0].capitalize()
            messages.append({
                "role": "user",
                "content": f"[{speaker_label}]: {content}"
            })

        # If no history yet, include the provocation as the prompt
        if not dialogue_history:
            messages.append({
                "role": "user",
                "content": f"Opening question for the circle:\n\n{provocation}\n\nPlease share your perspective."
            })
        else:
            # Prompt for continuation
            messages.append({
                "role": "user",
                "content": "Please respond to the dialogue above, staying in character and building on what others have shared."
            })

        return messages

    def _build_system_prompt(self, agent: Agent, provocation: str) -> str:
        """Build the system prompt for an agent."""
        other_agents = [a.name.split('-')[0].capitalize()
                       for aid, a in self.agents.items() if aid != agent.id]

        return f"""{agent.system_prompt}

You are participating in a Socratic dialogue circle exploring this question:

"{provocation}"

Other voices in this circle: {', '.join(other_agents)}

Guidelines:
- Stay in character throughout your response
- Build on what others have shared rather than repeating
- You may address specific participants by name
- Keep responses focused (2-4 paragraphs)
- Bring your unique epistemic lens to the conversation"""


# Convenience function
def run_session(
    config_path: Path,
    provocation: str,
    output_dir: Path,
    seed: int = 42,
    **kwargs
) -> Path:
    """
    Convenience function to run a dialogue session.

    Args:
        config_path: Path to ensemble config YAML
        provocation: Opening provocation text
        output_dir: Directory for output
        seed: Random seed
        **kwargs: Additional arguments for run_dialogue

    Returns:
        Path to saved session JSON
    """
    config = EnsembleConfig.from_yaml(config_path)
    orchestrator = DialogueOrchestrator(config)
    return orchestrator.run_dialogue(
        provocation=provocation,
        output_dir=output_dir,
        seed=seed,
        config_path=str(config_path),
        **kwargs
    )


# Test/demo if run directly
if __name__ == "__main__":
    import sys

    print("Orchestrator Test")
    print("=" * 50)

    # Check Ollama
    if not OllamaClient.is_running():
        print("Error: Ollama not running. Start with: ollama serve")
        sys.exit(1)

    # Load config
    config_path = Path("experiments/config/multi_model.yaml")
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    config = EnsembleConfig.from_yaml(config_path)
    print(f"Config loaded: {config.mode}")

    # Validate models
    required = list(set(config.get_model_for_agent(aid) for aid in config.agents))
    validation = OllamaClient.validate_models(required)
    missing = [m for m, available in validation.items() if not available]

    if missing:
        print(f"Error: Missing models: {missing}")
        print("Install with: ollama pull <model>")
        sys.exit(1)

    print("All required models available")

    # Run short test
    orchestrator = DialogueOrchestrator(config)
    output_dir = Path("experiments/runs")

    print("\nRunning 3-turn test dialogue...\n")

    session_path = orchestrator.run_dialogue(
        provocation="What does it mean to truly listen?",
        output_dir=output_dir,
        max_turns=3,
        seed=42,
        compute_embeddings=True
    )

    print(f"\nTest complete. Session saved to: {session_path}")
