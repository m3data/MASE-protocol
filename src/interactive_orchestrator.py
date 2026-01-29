"""
Interactive dialogue orchestration for MASE with human participation.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Wraps existing orchestrator infrastructure for real-time interactive
dialogue with human as the 8th participant in the circle.

Architecture: Queue-based event streaming
- Dialogue loop runs in background thread
- Events pushed to thread-safe queue
- SSE endpoint reads from queue (survives reconnection)
- Human input submitted via REST, picked up by worker thread
"""

import re
import random
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Generator, Tuple, Union
from enum import Enum


def strip_voice_bleed(content: str, agent_id: str) -> str:
    """
    Remove self-labeling prefixes from agent responses.

    Patterns removed:
    - "AgentName: ..."
    - "As AgentName, ..."
    - "As AgentName I ..."
    - "AgentName here. ..."
    """
    name = agent_id.capitalize()

    # Pattern: "Name: " at start
    content = re.sub(rf'^\s*{name}:\s*', '', content, flags=re.IGNORECASE)

    # Pattern: "As Name, " or "As Name I" at start
    content = re.sub(rf'^\s*As {name}[,:]?\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(rf'^\s*As {name} I\s+', 'I ', content, flags=re.IGNORECASE)

    # Pattern: "Name here." at start
    content = re.sub(rf'^\s*{name} here[.,]?\s*', '', content, flags=re.IGNORECASE)

    # Pattern: "I would respond:" or similar meta-commentary
    content = re.sub(r'^\s*I would respond:\s*', '', content, flags=re.IGNORECASE)

    return content.strip()

# Handle both package and direct execution
try:
    from .ollama_client import OllamaClient
    from .agents import Agent, EnsembleConfig, load_ensemble
    from .session_logger import SessionLogger, TurnRecord
    from .session_analysis import SessionAnalyzer
except ImportError:
    from ollama_client import OllamaClient
    from agents import Agent, EnsembleConfig, load_ensemble
    from session_logger import SessionLogger, TurnRecord
    from session_analysis import SessionAnalyzer


class SessionState(Enum):
    """Possible states for an interactive session."""
    IDLE = "idle"              # Not started
    RUNNING = "running"        # AI agents conversing
    PAUSED = "paused"          # Manually paused
    AWAITING_HUMAN = "awaiting_human"  # Human's turn
    COMPLETE = "complete"      # Session ended


@dataclass
class TurnEvent:
    """Event emitted when a turn is generated."""
    turn_number: int
    agent_id: str
    agent_name: str
    content: str
    model: str
    latency_ms: float
    is_human: bool = False


@dataclass
class StateEvent:
    """Event emitted when session state changes."""
    state: SessionState
    next_speaker: Optional[str] = None
    message: Optional[str] = None


@dataclass
class MetricsEvent:
    """Event emitted with live metrics during dialogue (every N turns)."""
    turn_number: int
    basin: str
    basin_confidence: float
    integrity_score: float
    integrity_label: str
    psi_semantic: float
    psi_temporal: float
    psi_affective: float
    voice_distinctiveness: float
    velocity_magnitude: Optional[float] = None
    type: str = "metrics"


class InteractiveTurnSelector:
    """
    Turn selector that includes human as a participant.

    Extends the logic from TurnSelector to support:
    - Human as 8th agent in rotation
    - Cooldown period to prevent any voice from dominating
    - Mention detection for human ("you", "human", etc.)
    - Force-invoke for specific agents
    """

    HUMAN_ID = "human"
    HUMAN_MENTIONS = ["@human", "@you", "@mat"]  # @-prefixed for explicit mentions
    HUMAN_MENTIONS_LOOSE = ["you", "human", "mat", "your"]  # Loose matching (lower priority)

    def __init__(self, agents: Dict[str, Agent], seed: int, include_human: bool = True, cooldown: int = 2):
        """
        Initialize turn selector.

        Args:
            agents: Dict of AI agents
            seed: Random seed for reproducibility
            include_human: Whether to include human in rotation
            cooldown: Number of turns an agent must wait before speaking again
        """
        self.agents = agents
        self.include_human = include_human
        self.rng = random.Random(seed)
        self.cooldown = cooldown

        # Build agent list (AI agents + optionally human)
        self.agent_ids = list(agents.keys())
        if include_human:
            self.agent_ids.append(self.HUMAN_ID)

        # Track turn counts for balancing
        self.turn_counts: Dict[str, int] = {aid: 0 for aid in self.agent_ids}
        # Track recent speakers for cooldown (most recent last)
        self.recent_speakers: List[str] = []

    def select_next(
        self,
        last_content: Optional[str] = None,
        force_agent: Optional[str] = None
    ) -> str:
        """
        Select the next speaker.

        Args:
            last_content: Content of the last turn (for mention detection)
            force_agent: Force a specific agent to speak

        Returns:
            Agent ID of next speaker (may be "human")
        """
        if force_agent and force_agent in self.agent_ids:
            return self._select(force_agent)

        # Get agents in cooldown (most recent N speakers)
        in_cooldown = set(self.recent_speakers[-self.cooldown:]) if self.cooldown > 0 else set()

        # Get eligible agents (exclude those in cooldown)
        eligible = [aid for aid in self.agent_ids if aid not in in_cooldown]

        if not eligible:
            # Fallback: allow all if cooldown excludes everyone (small ensembles)
            eligible = self.agent_ids

        # Check for mentions in last content
        if last_content:
            mentioned = self._detect_mentions(last_content)
            mentioned_eligible = [aid for aid in mentioned if aid in eligible]
            if mentioned_eligible:
                return self._select(mentioned_eligible[0])

        # Weighted random selection
        return self._select(self._weighted_choice(eligible))

    def _detect_mentions(self, content: str) -> List[str]:
        """
        Detect agent or human mentions in content.

        Supports two mention styles:
        - Explicit @mentions: @Luma, @Orin, @Human (high priority)
        - Loose mentions: bare names in text (lower priority)

        Returns list of agent IDs, with @mentioned agents first.
        """
        content_lower = content.lower()
        explicit_mentions = []  # @-prefixed mentions (high priority)
        loose_mentions = []     # Bare name mentions (lower priority)

        # Check for explicit @mentions first (highest priority)
        # Pattern: @name where name is an agent or human
        at_mentions = re.findall(r'@(\w+)', content_lower)

        for mention in at_mentions:
            # Check if it's human
            if self.include_human and mention in ['human', 'you', 'mat']:
                if self.HUMAN_ID not in explicit_mentions:
                    explicit_mentions.append(self.HUMAN_ID)
            # Check if it's an agent
            for agent_id in self.agents:
                if mention == agent_id.lower():
                    if agent_id not in explicit_mentions:
                        explicit_mentions.append(agent_id)
                    break

        # Check for loose human mentions (lower priority)
        if self.include_human and self.HUMAN_ID not in explicit_mentions:
            for trigger in self.HUMAN_MENTIONS_LOOSE:
                if trigger in content_lower:
                    if self.HUMAN_ID not in loose_mentions:
                        loose_mentions.append(self.HUMAN_ID)
                    break

        # Check for loose AI agent mentions (lower priority)
        for agent_id, agent in self.agents.items():
            if agent_id in explicit_mentions:
                continue  # Already captured as explicit
            if agent_id.lower() in content_lower:
                if agent_id not in loose_mentions:
                    loose_mentions.append(agent_id)
            elif agent.name and agent.name.split('-')[0].lower() in content_lower:
                if agent_id not in loose_mentions:
                    loose_mentions.append(agent_id)

        # Return explicit mentions first, then loose mentions
        return explicit_mentions + loose_mentions

    def _weighted_choice(self, eligible: List[str]) -> str:
        """Choose agent with weights inversely proportional to turn count."""
        if not eligible:
            return self.rng.choice(self.agent_ids)

        max_turns = max(self.turn_counts.values()) + 1
        weights = []
        for aid in eligible:
            weight = max_turns - self.turn_counts[aid] + 1
            weights.append(weight)

        total = sum(weights)
        r = self.rng.random() * total
        cumulative = 0
        for aid, weight in zip(eligible, weights):
            cumulative += weight
            if r <= cumulative:
                return aid

        return eligible[-1]

    def _select(self, agent_id: str) -> str:
        """Record selection and return agent ID."""
        self.turn_counts[agent_id] += 1
        self.recent_speakers.append(agent_id)
        return agent_id

    def record_interjection(self, agent_id: str):
        """Record an interjection (out-of-turn contribution)."""
        self.turn_counts[agent_id] += 1
        self.recent_speakers.append(agent_id)


class InteractiveSession:
    """
    Manages an interactive dialogue session with human participation.

    Uses a queue-based architecture:
    - Background thread runs dialogue loop
    - Events pushed to thread-safe queue
    - SSE reads from queue (survives reconnection)
    - Human input via REST triggers thread to continue
    """

    def __init__(
        self,
        config: EnsembleConfig,
        provocation: str,
        output_dir: Path,
        seed: int = 42,
        agents_dir: Optional[Path] = None,
        ollama_base_url: str = "http://localhost:11434",
        context_window: int = 5,
        opening_agent: Optional[str] = None,
        max_turns: int = 100
    ):
        """
        Initialize interactive session.

        Args:
            config: Ensemble configuration
            provocation: Opening provocation text
            output_dir: Directory for session output
            seed: Random seed
            agents_dir: Path to agent definitions
            ollama_base_url: Ollama server URL
            context_window: Number of recent turns in context
            opening_agent: First agent to speak
            max_turns: Maximum turns before auto-complete
        """
        self.config = config
        self.provocation = provocation
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.context_window = context_window
        self.opening_agent = opening_agent or config.dialogue_opening_agent
        self.max_turns = max_turns

        # Load AI agents
        self.agents = load_ensemble(agents_dir, config)

        # Initialize Ollama client
        self.ollama = OllamaClient(base_url=ollama_base_url)

        # Initialize turn selector with human
        self.turn_selector = InteractiveTurnSelector(
            self.agents, seed, include_human=True
        )

        # Build model/temperature assignments
        self.model_assignments = {
            aid: config.get_model_for_agent(aid)
            for aid in self.agents
        }
        self.model_assignments["human"] = "human"

        self.temp_assignments = {
            aid: config.get_temperature_for_agent(aid)
            for aid in self.agents
        }
        self.temp_assignments["human"] = 0.0

        # Session state
        self.state = SessionState.IDLE
        self.dialogue_history: List[Tuple[str, str, str]] = []  # (agent_id, name, content)
        self.turn_number = 0
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Queue-based event streaming
        self._event_queue: queue.Queue = queue.Queue()
        self._human_input_event: threading.Event = threading.Event()
        self._human_input_content: Optional[str] = None
        self._pending_invoke: Optional[str] = None
        self._stop_event: threading.Event = threading.Event()
        self._pause_event: threading.Event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._started: bool = False

        # Logger
        self.logger: Optional[SessionLogger] = None

        # Live metrics analyzer
        self._live_analyzer: Optional[SessionAnalyzer] = None
        self.metrics_interval: int = 3  # Emit metrics every N turns

        # Interjections (researcher prompts injected during dialogue)
        self.interjections: List[Dict] = []

    def get_agents_metadata(self) -> List[Dict]:
        """Get metadata for all agents (for frontend display)."""
        metadata = []

        for agent_id, agent in self.agents.items():
            metadata.append({
                "id": agent_id,
                "name": agent.name.split('-')[0].capitalize() if agent.name else agent_id.capitalize(),
                "full_name": agent.name,
                "model": self.model_assignments.get(agent_id, "unknown"),
                "temperature": self.temp_assignments.get(agent_id, 0.7),
                "is_human": False
            })

        # Add human
        metadata.append({
            "id": "human",
            "name": "You",
            "full_name": "human-participant",
            "model": "human",
            "temperature": 0.0,
            "is_human": True
        })

        return metadata

    def get_state(self) -> Dict:
        """Get current session state."""
        return {
            "state": self.state.value,
            "turn_number": self.turn_number,
            "next_speaker": None,  # Determined by worker thread
            "session_id": self.session_id,
            "history_length": len(self.dialogue_history)
        }

    def compute_live_metrics(self) -> Optional[MetricsEvent]:
        """
        Compute live metrics from current dialogue state.

        Uses the SessionAnalyzer incrementally to compute basin detection,
        integrity, and Ψ vector components.

        Returns:
            MetricsEvent with current metrics, or None if insufficient data
        """
        if not self._live_analyzer or len(self.dialogue_history) < 4:
            return None

        try:
            # Get summary from analyzer
            summary = self._live_analyzer.get_summary(compute_ci=False)

            # Get latest turn state for Ψ components
            latest_state = summary.turn_states[-1] if summary.turn_states else None

            return MetricsEvent(
                turn_number=self.turn_number,
                basin=summary.dominant_basin,
                basin_confidence=summary.dominant_basin_percentage,
                integrity_score=summary.integrity_score or 0.0,
                integrity_label=summary.integrity_label or 'unknown',
                psi_semantic=latest_state.psi_semantic if latest_state else 0.0,
                psi_temporal=latest_state.psi_temporal if latest_state else 0.0,
                psi_affective=latest_state.psi_affective if latest_state else 0.0,
                voice_distinctiveness=summary.voice_distinctiveness,
                velocity_magnitude=latest_state.velocity_magnitude if latest_state else None
            )
        except Exception as e:
            print(f"[WARN] Failed to compute live metrics: {e}")
            return None

    def inject_prompt(self, content: str, trigger_response: bool = True) -> None:
        """
        Add a researcher interjection to the dialogue context.

        The prompt is NOT counted as a turn but is visible to agents
        in subsequent context windows. Logged separately in session JSON.

        Args:
            content: The researcher's prompt to inject
            trigger_response: If True, trigger an agent to respond after injection
        """
        interjection = {
            'turn': self.turn_number,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.interjections.append(interjection)

        # Add to dialogue history with special marker (agents will see it)
        self.dialogue_history.append(
            ("researcher", "Researcher", f"[Interjection]: {content}")
        )

        # Log if logger available
        if self.logger:
            self.logger.log_turn(
                agent_id="researcher",
                agent_name="Researcher",
                content=f"[Interjection]: {content}",
                model="n/a",
                temperature=0.0,
                latency_ms=0.0,
                embedding=None,
                prompt_tokens=None,
                completion_tokens=None
            )

        # Queue event so frontend can display the injection
        injection_event = TurnEvent(
            turn_number=self.turn_number,
            agent_id="researcher",
            agent_name="Researcher",
            content=content,
            model="n/a",
            latency_ms=0.0,
            is_human=False
        )
        self._event_queue.put(injection_event)

        # Trigger agent response if requested
        if trigger_response:
            self._trigger_next_response()

    def start(self) -> "InteractiveSession":
        """
        Start the dialogue session.

        Launches background worker thread and returns self for event iteration.
        Call get_next_event() to receive events, or iterate over the session.
        """
        if self._started:
            return self  # Already started, just return for continued iteration

        self._started = True
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize live metrics analyzer
        self._live_analyzer = SessionAnalyzer()

        # Initialize logger with our pre-generated session_id
        self.logger = SessionLogger(self.output_dir, session_id=self.session_id)
        self.logger.start_session(
            mode=f"{self.config.mode}_interactive",
            provocation_text=self.provocation,
            seed=self.seed,
            model_assignments=self.model_assignments,
            temperature_assignments=self.temp_assignments,
            provocation_id=None,
            config_path=None
        )

        # Start the worker thread
        self._worker_thread = threading.Thread(
            target=self._dialogue_loop,
            daemon=True,
            name=f"mase-session-{self.session_id}"
        )
        self._worker_thread.start()

        return self

    def get_next_event(self, timeout: float = 30.0) -> Optional[Union[TurnEvent, StateEvent]]:
        """
        Get the next event from the queue.

        Args:
            timeout: How long to wait for an event (seconds)

        Returns:
            Event or None if timeout/stopped
        """
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def has_events(self) -> bool:
        """Check if there are pending events in the queue."""
        return not self._event_queue.empty()

    def _dialogue_loop(self):
        """
        Background thread: runs the dialogue, pushing events to queue.
        Blocks when awaiting human input.
        """
        try:
            self._run_dialogue()
        except Exception as e:
            import traceback
            error_msg = f"Worker thread error: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self._event_queue.put(StateEvent(
                state=SessionState.COMPLETE,
                message=f"Error: {str(e)}"
            ))
            self.state = SessionState.COMPLETE

    def _run_dialogue(self):
        """Inner dialogue loop - separated for clean error handling."""
        self.state = SessionState.RUNNING
        self._event_queue.put(StateEvent(state=self.state, message="Session started"))

        while self.turn_number < self.max_turns and not self._stop_event.is_set():
            # Check for pause
            if self._pause_event.is_set():
                self.state = SessionState.PAUSED
                self._event_queue.put(StateEvent(state=self.state, message="Paused"))
                # Wait until unpaused or stopped
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    time.sleep(0.1)
                if self._stop_event.is_set():
                    break
                self.state = SessionState.RUNNING
                self._event_queue.put(StateEvent(state=self.state, message="Resumed"))

            # Select next speaker
            last_content = self.dialogue_history[-1][2] if self.dialogue_history else None
            force = self.opening_agent if self.turn_number == 0 else self._pending_invoke
            self._pending_invoke = None

            next_speaker = self.turn_selector.select_next(last_content, force)

            if next_speaker == "human":
                # Human's turn - signal and wait for input
                print(f"[DEBUG] Worker: human's turn, waiting for input...", flush=True)
                self.state = SessionState.AWAITING_HUMAN
                self._human_input_event.clear()
                self._event_queue.put(StateEvent(
                    state=self.state,
                    next_speaker="human",
                    message="Your turn to speak"
                ))

                # Block until human input received or stopped
                while not self._human_input_event.is_set() and not self._stop_event.is_set():
                    time.sleep(0.1)

                if self._stop_event.is_set():
                    print("[DEBUG] Worker: stop event received", flush=True)
                    break

                # Process human input (already added to history by submit_human_turn)
                print(f"[DEBUG] Worker: human input received, continuing...", flush=True)
                self._human_input_event.clear()
                self.state = SessionState.RUNNING
                continue

            # AI agent's turn - signal who's speaking before generating
            self._event_queue.put(StateEvent(
                state=SessionState.RUNNING,
                next_speaker=next_speaker,
                message=f"{next_speaker.capitalize()} is thinking..."
            ))

            turn_event = self._generate_ai_turn(next_speaker)
            self._event_queue.put(turn_event)
            self.turn_number += 1

            # Emit metrics every N turns
            if self.turn_number > 0 and self.turn_number % self.metrics_interval == 0:
                metrics_event = self.compute_live_metrics()
                if metrics_event:
                    self._event_queue.put(metrics_event)

        # Session complete
        if not self._stop_event.is_set():
            self.state = SessionState.COMPLETE
            self._finalize()
            self._event_queue.put(StateEvent(state=self.state, message="Session complete"))

    def pause(self):
        """Pause the session."""
        if self.state == SessionState.RUNNING:
            self._pause_event.set()

    def resume(self):
        """Resume a paused session."""
        if self.state == SessionState.PAUSED:
            self._pause_event.clear()

    def submit_human_turn(self, content: str) -> TurnEvent:
        """
        Submit human's contribution to the dialogue.

        Args:
            content: Human's message

        Returns:
            TurnEvent for the human turn
        """
        # Check if worker thread is still running
        if self._worker_thread and not self._worker_thread.is_alive():
            print(f"[WARN] Worker thread dead, restarting...")
            self._started = False
            self._stop_event.clear()
            self.start()

        self.turn_number += 1
        print(f"[DEBUG] submit_human_turn: turn={self.turn_number}, state={self.state}, thread_alive={self._worker_thread.is_alive() if self._worker_thread else 'None'}", flush=True)

        # Log the turn
        if self.logger:
            self.logger.log_turn(
                agent_id="human",
                agent_name="human-participant",
                content=content,
                model="human",
                temperature=0.0,
                latency_ms=0.0,
                embedding=None,
                prompt_tokens=None,
                completion_tokens=None
            )

        # Update history
        self.dialogue_history.append(("human", "human-participant", content))

        # Process turn through live analyzer
        if self._live_analyzer:
            try:
                self._live_analyzer.process_turn(
                    content=content,
                    agent_id="human",
                    embedding=None
                )
            except Exception as e:
                print(f"[WARN] Live analyzer error for human turn: {e}")

        # Record in turn selector
        self.turn_selector.record_interjection("human")

        # Create turn event
        turn_event = TurnEvent(
            turn_number=self.turn_number,
            agent_id="human",
            agent_name="You",
            content=content,
            model="human",
            latency_ms=0.0,
            is_human=True
        )

        # Push turn event to queue so SSE can send it
        self._event_queue.put(turn_event)
        print(f"[DEBUG] Human turn queued, queue size={self._event_queue.qsize()}", flush=True)

        # Update state and signal worker thread
        was_awaiting = self.state == SessionState.AWAITING_HUMAN
        self.state = SessionState.RUNNING

        # Emit state change event so frontend knows state changed
        self._event_queue.put(StateEvent(
            state=self.state,
            next_speaker=None,
            message="Human turn submitted"
        ))

        # Signal worker thread to continue
        self._human_input_event.set()
        print(f"[DEBUG] Signaled worker thread, was_awaiting={was_awaiting}, human_input_event={self._human_input_event.is_set()}", flush=True)

        return turn_event

    def invoke_agent(self, agent_id: str):
        """Request a specific agent to speak next."""
        if agent_id in self.agents or agent_id == "human":
            self._pending_invoke = agent_id
            self._trigger_next_response()

    def _trigger_next_response(self):
        """
        Signal the worker thread to generate the next response.

        Handles various session states:
        - AWAITING_HUMAN: Signal to continue (AI will respond)
        - PAUSED: Resume the session
        - RUNNING: No action needed, worker will continue
        """
        if self.state == SessionState.AWAITING_HUMAN:
            # Signal worker to continue even though human didn't speak
            self._human_input_event.set()
        elif self.state == SessionState.PAUSED:
            # Resume the session
            self._pause_event.clear()

    def end_session(self) -> Optional[Path]:
        """End the session and save."""
        self._stop_event.set()
        self._human_input_event.set()  # Unblock if waiting
        self._pause_event.clear()  # Unblock if paused

        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        self.state = SessionState.COMPLETE
        return self._finalize()

    def _generate_ai_turn(self, agent_id: str) -> TurnEvent:
        """Generate a turn for an AI agent."""
        agent = self.agents[agent_id]

        # Build context
        context = self._build_context(agent)

        # Get personality-derived sampling params if available
        extra_options = None
        temperature = agent.temperature
        if agent.personality:
            personality_params = agent.personality.to_sampling_params()
            temperature = personality_params.pop('temperature', agent.temperature)
            extra_options = personality_params if personality_params else None

        # Generate response
        start_time = datetime.now()
        response_text, metadata = self.ollama.generate(
            model=agent.model,
            messages=context,
            temperature=temperature,
            seed=self.seed + self.turn_number,
            extra_options=extra_options
        )
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Strip voice bleed (self-labeling prefixes)
        response_text = strip_voice_bleed(response_text, agent_id)

        # Log the turn
        if self.logger:
            self.logger.log_turn(
                agent_id=agent_id,
                agent_name=agent.name,
                content=response_text,
                model=agent.model,
                temperature=agent.temperature,
                latency_ms=latency_ms,
                embedding=None,
                prompt_tokens=metadata.prompt_tokens,
                completion_tokens=metadata.completion_tokens
            )

        # Update history
        self.dialogue_history.append((agent_id, agent.name, response_text))

        # Process turn through live analyzer (without embedding for speed)
        if self._live_analyzer:
            try:
                self._live_analyzer.process_turn(
                    content=response_text,
                    agent_id=agent_id,
                    embedding=None  # Skip embedding for live performance
                )
            except Exception as e:
                print(f"[WARN] Live analyzer error: {e}")

        # Use agent_id as display name (e.g., "Orin" not "Systems")
        display_name = agent_id.capitalize()

        return TurnEvent(
            turn_number=self.turn_number + 1,
            agent_id=agent_id,
            agent_name=display_name,
            content=response_text,
            model=agent.model,
            latency_ms=latency_ms,
            is_human=False
        )

    def _build_context(self, agent: Agent) -> List[Dict[str, str]]:
        """Build the message context for an agent."""
        messages = []

        # System prompt
        other_names = [
            a.name.split('-')[0].capitalize()
            for aid, a in self.agents.items()
            if aid != agent.id
        ]
        other_names.append("You (the human participant)")

        # Build personality description if available
        personality_desc = ""
        if agent.personality:
            desc = agent.personality.to_prompt_description()
            if desc:
                personality_desc = f"\n\n{desc}"

        system_prompt = f"""{agent.system_prompt}{personality_desc}

You are participating in a Socratic dialogue circle exploring: "{self.provocation}"

Other voices: {', '.join(other_names)}

ADDRESSING OTHERS:
- Use @Name to directly address someone (e.g., @Luma, @Orin, @Human)
- When you @mention someone, they will respond next
- If someone @mentions you, respond to their specific point
- Use @mentions sparingly - only when you genuinely want that voice's perspective

CRITICAL RULES:
- Never prefix your response with your name or "As [name]" - the system identifies speakers
- You are {agent.id.upper()} only. NEVER speak as or pretend to be another participant.
- Keep responses SHORT: 2-3 sentences maximum, occasionally up to a short paragraph.
- Be direct and concise. This is a conversation, not an essay.
- Build on what others said, don't summarize or repeat.
- Match the tone and energy of the provocation before applying your analytical lens."""

        messages.append({"role": "system", "content": system_prompt})

        # Recent dialogue
        recent = self.dialogue_history[-self.context_window:] if self.dialogue_history else []

        for agent_id, agent_name, content in recent:
            if agent_id == "human":
                speaker_label = "Human"
            else:
                speaker_label = agent_name.split('-')[0].capitalize() if agent_name else agent_id

            messages.append({
                "role": "user",
                "content": f"[{speaker_label}]: {content}"
            })

        # Prompt
        if not self.dialogue_history:
            messages.append({
                "role": "user",
                "content": f"Opening question: {self.provocation}\n\nShare your perspective briefly (2-3 sentences)."
            })
        else:
            messages.append({
                "role": "user",
                "content": "Respond briefly (2-3 sentences). Speak only as yourself, never as other participants."
            })

        return messages

    def _finalize(self) -> Optional[Path]:
        """Finalize and save the session."""
        if self.logger:
            return self.logger.end_session()
        return None


# Convenience function for testing
def create_interactive_session(
    provocation: str,
    output_dir: str = "sessions",
    config_path: Optional[str] = None,
    seed: int = 42
) -> InteractiveSession:
    """
    Create an interactive session with default settings.

    Args:
        provocation: Opening question
        output_dir: Where to save sessions
        config_path: Path to config YAML (default: multi_model.yaml)
        seed: Random seed

    Returns:
        Configured InteractiveSession
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "experiments" / "config" / "multi_model.yaml"
    else:
        config_path = Path(config_path)

    config = EnsembleConfig.from_yaml(config_path)

    return InteractiveSession(
        config=config,
        provocation=provocation,
        output_dir=Path(output_dir),
        seed=seed
    )
