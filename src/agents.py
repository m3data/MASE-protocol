"""
Agent loading and ensemble configuration for MASE.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Loads agent definitions from .claude/agents/*.md files and maps them
to models based on experiment configuration.
"""

import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Agent:
    """Configuration for a MASE agent."""
    id: str                          # Short identifier (e.g., "luma", "elowen")
    name: str                        # Full name from frontmatter
    system_prompt: str               # The agent's prompt/persona
    model: Optional[str] = None      # Assigned model (from ensemble config)
    temperature: float = 0.7         # Generation temperature


@dataclass
class AgentConfig:
    """Per-agent configuration from experiment YAML."""
    model: str
    temperature: float = 0.7


@dataclass
class EnsembleConfig:
    """
    Configuration for an ensemble of agents.

    Can be loaded from YAML or constructed programmatically.
    """
    mode: str                                    # "single_model" or "multi_model"
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    shared_model: Optional[str] = None           # For single_model mode
    dialogue_max_turns: int = 21
    dialogue_context_window: int = 5
    dialogue_opening_agent: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "EnsembleConfig":
        """Load ensemble configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        mode = data.get("mode", "single_model")
        shared_model = data.get("shared_model")

        # Parse agent configs
        agents = {}
        for agent_id, config in data.get("agents", {}).items():
            if isinstance(config, dict):
                agents[agent_id] = AgentConfig(
                    model=config.get("model", shared_model),
                    temperature=config.get("temperature", 0.7)
                )
            else:
                # Simple format: agent_id: model_name
                agents[agent_id] = AgentConfig(model=config)

        # Parse dialogue settings
        dialogue = data.get("dialogue", {})

        return cls(
            mode=mode,
            agents=agents,
            shared_model=shared_model,
            dialogue_max_turns=dialogue.get("max_turns", 21),
            dialogue_context_window=dialogue.get("context_window", 5),
            dialogue_opening_agent=dialogue.get("opening_agent")
        )

    def get_model_for_agent(self, agent_id: str) -> Optional[str]:
        """Get the model assigned to an agent."""
        if agent_id in self.agents:
            return self.agents[agent_id].model
        return self.shared_model

    def get_temperature_for_agent(self, agent_id: str) -> float:
        """Get the temperature for an agent."""
        if agent_id in self.agents:
            return self.agents[agent_id].temperature
        return 0.7


class AgentLoader:
    """
    Loads agent definitions from .claude/agents/ directory.

    Parses YAML frontmatter and extracts system prompts from markdown files.
    """

    # Map from filename to short agent ID
    AGENT_ID_MAP = {
        "luma-child-voice": "luma",
        "elowen-ecological-wisdom": "elowen",
        "systems-analyst-orin": "orin",
        "moral-imagination-explorer": "nyra",
        "ilya-liminal-guide": "ilya",
        "policy-pragmatist-sefi": "sefi",
        "capitalist-realist-tala": "tala",
    }

    def __init__(self, agents_dir: Optional[Path] = None):
        """
        Initialize agent loader.

        Args:
            agents_dir: Path to agents directory. Defaults to MASE/.claude/agents/
        """
        if agents_dir is None:
            # Default to MASE/.claude/agents/ relative to this file
            self.agents_dir = Path(__file__).parent.parent / ".claude" / "agents"
        else:
            self.agents_dir = Path(agents_dir)

        self._agents: Dict[str, Agent] = {}

    def load_all(self) -> Dict[str, Agent]:
        """
        Load all agents from the agents directory.

        Returns:
            Dict mapping agent ID to Agent object
        """
        if not self.agents_dir.exists():
            raise FileNotFoundError(f"Agents directory not found: {self.agents_dir}")

        for md_file in self.agents_dir.glob("*.md"):
            agent = self._parse_agent_file(md_file)
            if agent:
                self._agents[agent.id] = agent

        return self._agents

    def _parse_agent_file(self, path: Path) -> Optional[Agent]:
        """Parse a single agent markdown file."""
        content = path.read_text()

        # Extract YAML frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---\n(.*)$', content, re.DOTALL)
        if not frontmatter_match:
            return None

        frontmatter_str = frontmatter_match.group(1)
        body = frontmatter_match.group(2).strip()

        # Parse frontmatter - handle complex description fields with colons
        # by extracting just the fields we need via regex
        name_match = re.search(r'^name:\s*(.+)$', frontmatter_str, re.MULTILINE)
        model_match = re.search(r'^model:\s*(.+)$', frontmatter_str, re.MULTILINE)

        name = name_match.group(1).strip() if name_match else path.stem

        # Determine agent ID from filename
        filename_stem = path.stem
        agent_id = self.AGENT_ID_MAP.get(filename_stem, filename_stem)

        return Agent(
            id=agent_id,
            name=name,
            system_prompt=body
        )

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get a specific agent by ID."""
        if not self._agents:
            self.load_all()
        return self._agents.get(agent_id)

    def get_all_agent_ids(self) -> List[str]:
        """Get list of all agent IDs."""
        if not self._agents:
            self.load_all()
        return list(self._agents.keys())


def load_ensemble(
    agents_dir: Optional[Path] = None,
    config: Optional[EnsembleConfig] = None
) -> Dict[str, Agent]:
    """
    Load agents and apply ensemble configuration.

    Args:
        agents_dir: Path to agents directory
        config: Ensemble configuration (model mapping)

    Returns:
        Dict of Agent objects with model assignments applied
    """
    loader = AgentLoader(agents_dir)
    agents = loader.load_all()

    if config:
        for agent_id, agent in agents.items():
            agent.model = config.get_model_for_agent(agent_id)
            agent.temperature = config.get_temperature_for_agent(agent_id)

    return agents


# Test if run directly
if __name__ == "__main__":
    print("Agent Loader Test")
    print("=" * 50)

    loader = AgentLoader()

    try:
        agents = loader.load_all()
        print(f"\nLoaded {len(agents)} agents:")

        for agent_id, agent in agents.items():
            print(f"\n  {agent_id}:")
            print(f"    name: {agent.name}")
            print(f"    prompt length: {len(agent.system_prompt)} chars")
            print(f"    first line: {agent.system_prompt.split(chr(10))[0][:60]}...")

    except FileNotFoundError as e:
        print(f"Error: {e}")
