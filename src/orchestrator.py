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
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from .ollama_client import OllamaClient
from .agents import Agent, EnsembleConfig, load_ensemble
from .embedding_service import EmbeddingService
from .session_logger import SessionLogger, TurnRecord


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
    """

    def __init__(
        self,
        config: EnsembleConfig,
        agents_dir: Optional[Path] = None,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize orchestrator.

        Args:
            config: Ensemble configuration
            agents_dir: Path to agent definitions
            ollama_base_url: Ollama server URL
        """
        self.config = config
        self.agents = load_ensemble(agents_dir, config)
        self.ollama = OllamaClient(base_url=ollama_base_url)
        self.embedding_service: Optional[EmbeddingService] = None

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
        context_window: Optional[int] = None
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

        # Start session
        logger.start_session(
            mode=self.config.mode,
            provocation_text=provocation,
            seed=seed,
            model_assignments=model_assignments,
            temperature_assignments=temp_assignments,
            provocation_id=provocation_id,
            config_path=config_path
        )

        # Track dialogue history for context building
        dialogue_history: List[Tuple[str, str, str]] = []  # (agent_id, agent_name, content)

        print(f"\n{'='*60}")
        print(f"MASE Dialogue Session")
        print(f"Mode: {self.config.mode}")
        print(f"Seed: {seed}")
        print(f"Max turns: {max_turns}")
        print(f"{'='*60}\n")
        print(f"Provocation:\n{provocation}\n")
        print(f"{'='*60}\n")

        # Main dialogue loop
        for turn_num in range(1, max_turns + 1):
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

            # Generate response
            print(f"[Turn {turn_num}] {agent_id} ({agent.model})...")

            response_text, metadata = self.ollama.generate(
                model=agent.model,
                messages=context,
                temperature=agent.temperature,
                seed=seed + turn_num  # Vary seed per turn for diversity
            )

            print(f"  Latency: {metadata.latency_ms:.0f}ms")
            print(f"  {agent_id}: {response_text[:100]}{'...' if len(response_text) > 100 else ''}\n")

            # Compute embedding
            embedding = None
            if compute_embeddings and self.embedding_service:
                embedding = self.embedding_service.embed(response_text)

            # Log turn
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

        # End session
        session_path = logger.end_session()

        print(f"{'='*60}")
        print(f"Session complete. Saved to: {session_path}")
        print(f"{'='*60}\n")

        return session_path

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
