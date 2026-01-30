"""
Agent loading and ensemble configuration for MASE.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Two-layer architecture:
- Templates: Epistemic lenses and voice archetypes (reusable patterns)
- Personas: Named instances with character details (reference templates)

Loads agent definitions from YAML files and composes system prompts
by layering template + persona content.
"""

import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# =============================================================================
# Personality (Big Five / OCEAN)
# =============================================================================

@dataclass
class Personality:
    """Big Five (OCEAN) personality traits, each 0.0-1.0."""
    openness: float = 0.5           # Curious/imaginative vs conventional/practical
    conscientiousness: float = 0.5  # Organized/disciplined vs flexible/spontaneous
    extraversion: float = 0.5       # Outgoing/energetic vs reserved/reflective
    agreeableness: float = 0.5      # Cooperative/trusting vs challenging/skeptical
    neuroticism: float = 0.5        # Sensitive/reactive vs stable/calm

    def to_prompt_description(self) -> str:
        """Generate natural language description of personality for system prompt."""
        traits = []

        # Openness
        if self.openness >= 0.7:
            traits.append("highly curious and imaginative, drawn to novel ideas")
        elif self.openness <= 0.3:
            traits.append("practical and grounded, preferring proven approaches")

        # Conscientiousness
        if self.conscientiousness >= 0.7:
            traits.append("precise and methodical, values structure and follow-through")
        elif self.conscientiousness <= 0.3:
            traits.append("spontaneous and flexible, comfortable with ambiguity")

        # Extraversion
        if self.extraversion >= 0.7:
            traits.append("energetic and expressive, thinks out loud")
        elif self.extraversion <= 0.3:
            traits.append("reflective and measured, speaks with deliberation")

        # Agreeableness
        if self.agreeableness >= 0.7:
            traits.append("warm and collaborative, seeks common ground")
        elif self.agreeableness <= 0.3:
            traits.append("direct and challenging, comfortable with friction")

        # Neuroticism
        if self.neuroticism >= 0.7:
            traits.append("emotionally attuned, responsive to tension and nuance")
        elif self.neuroticism <= 0.3:
            traits.append("steady and unflappable, maintains composure under pressure")

        if not traits:
            return ""

        return "Your personality: " + "; ".join(traits) + "."

    def to_sampling_params(self) -> Dict[str, float]:
        """Map personality traits to Ollama sampling parameters."""
        params = {}

        # Openness → temperature (higher = more creative/varied)
        # Map 0.0-1.0 to 0.4-1.0 temperature range
        params['temperature'] = 0.4 + (self.openness * 0.6)

        # Conscientiousness → top_p (higher C = lower top_p = more focused)
        # Map 0.0-1.0 to 0.95-0.7 top_p range
        params['top_p'] = 0.95 - (self.conscientiousness * 0.25)

        # Neuroticism → repeat_penalty (higher N = higher penalty = more varied/jumpy)
        # Map 0.0-1.0 to 1.0-1.3 repeat_penalty range
        params['repeat_penalty'] = 1.0 + (self.neuroticism * 0.3)

        return params

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "Personality":
        """Create Personality from dict."""
        return cls(
            openness=float(data.get('openness', 0.5)),
            conscientiousness=float(data.get('conscientiousness', 0.5)),
            extraversion=float(data.get('extraversion', 0.5)),
            agreeableness=float(data.get('agreeableness', 0.5)),
            neuroticism=float(data.get('neuroticism', 0.5))
        )

    def merge(self, overrides: Optional[Dict[str, float]]) -> "Personality":
        """Return new Personality with overrides applied."""
        if not overrides:
            return self
        return Personality(
            openness=float(overrides.get('openness', self.openness)),
            conscientiousness=float(overrides.get('conscientiousness', self.conscientiousness)),
            extraversion=float(overrides.get('extraversion', self.extraversion)),
            agreeableness=float(overrides.get('agreeableness', self.agreeableness)),
            neuroticism=float(overrides.get('neuroticism', self.neuroticism))
        )


# =============================================================================
# Template (Epistemic Lens / Voice Archetype)
# =============================================================================

@dataclass
class VoiceGuidance:
    """Voice characteristics for a template."""
    style: str = ""
    register: str = ""
    patterns: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "VoiceGuidance":
        if not data:
            return cls()
        return cls(
            style=data.get('style', ''),
            register=data.get('register', ''),
            patterns=data.get('patterns', []),
            avoid=data.get('avoid', [])
        )

    def to_prompt_section(self) -> str:
        """Generate voice guidance section for system prompt."""
        lines = []
        if self.style:
            lines.append(f"Voice style: {self.style}")
        if self.register:
            lines.append(f"Register: {self.register}")
        if self.patterns:
            lines.append("Voice patterns:")
            for p in self.patterns:
                lines.append(f"  - {p}")
        if self.avoid:
            lines.append("Avoid:")
            for a in self.avoid:
                lines.append(f"  - {a}")
        return "\n".join(lines)


@dataclass
class Template:
    """
    Epistemic lens / voice archetype.

    Defines how an agent thinks and speaks at the archetype level.
    Personas reference templates and add character-specific details.
    """
    id: str
    name: str
    description: str
    epistemic_lens: str
    voice_guidance: VoiceGuidance
    default_personality: Personality

    @classmethod
    def from_yaml(cls, path: Path) -> "Template":
        """Load template from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            epistemic_lens=data.get('epistemic_lens', ''),
            voice_guidance=VoiceGuidance.from_dict(data.get('voice_guidance')),
            default_personality=Personality.from_dict(data.get('default_personality', {}))
        )


# =============================================================================
# Persona (Named Character Instance)
# =============================================================================

@dataclass
class Persona:
    """
    Named character instance that references a template.

    Adds character details, personality overrides, signature phrases,
    and prompt additions on top of the template's epistemic lens.
    """
    id: str
    name: str
    template_id: str
    description: str
    color: str
    character: Dict[str, Any]
    personality_overrides: Optional[Dict[str, float]]
    signature_phrases: List[str]
    prompt_additions: str

    # Resolved template (populated by loader)
    template: Optional[Template] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Persona":
        """Load persona from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            id=data['id'],
            name=data['name'],
            template_id=data['template'],
            description=data.get('description', ''),
            color=data.get('color', '#888888'),
            character=data.get('character', {}),
            personality_overrides=data.get('personality'),
            signature_phrases=data.get('signature_phrases', []),
            prompt_additions=data.get('prompt_additions', ''),
            template=None  # Resolved later by loader
        )

    def get_personality(self) -> Personality:
        """Get merged personality (template defaults + persona overrides)."""
        if self.template:
            return self.template.default_personality.merge(self.personality_overrides)
        elif self.personality_overrides:
            return Personality.from_dict(self.personality_overrides)
        return Personality()


# =============================================================================
# Template Loader
# =============================================================================

class TemplateLoader:
    """Loads templates from agents/templates/ directory."""

    def __init__(self, templates_dir: Optional[Path] = None):
        if templates_dir is None:
            self.templates_dir = Path(__file__).parent.parent / "agents" / "templates"
        else:
            self.templates_dir = Path(templates_dir)
        self._templates: Dict[str, Template] = {}

    def load_all(self) -> Dict[str, Template]:
        """Load all templates."""
        if not self.templates_dir.exists():
            return {}

        for yaml_file in self.templates_dir.glob("*.yaml"):
            try:
                template = Template.from_yaml(yaml_file)
                self._templates[template.id] = template
            except Exception as e:
                print(f"[WARN] Failed to load template {yaml_file}: {e}")

        return self._templates

    def get(self, template_id: str) -> Optional[Template]:
        """Get a specific template."""
        if not self._templates:
            self.load_all()
        return self._templates.get(template_id)

    def list_all(self) -> List[Dict]:
        """List all templates with basic info."""
        if not self._templates:
            self.load_all()
        return [
            {"id": t.id, "name": t.name, "description": t.description}
            for t in self._templates.values()
        ]


# =============================================================================
# Persona Loader
# =============================================================================

class PersonaLoader:
    """Loads personas and resolves their template references."""

    def __init__(
        self,
        personas_dir: Optional[Path] = None,
        templates_dir: Optional[Path] = None
    ):
        if personas_dir is None:
            self.personas_dir = Path(__file__).parent.parent / "agents" / "personas"
        else:
            self.personas_dir = Path(personas_dir)

        self.template_loader = TemplateLoader(templates_dir)
        self._personas: Dict[str, Persona] = {}

    def load_all(self) -> Dict[str, Persona]:
        """Load all personas and resolve template references."""
        if not self.personas_dir.exists():
            return {}

        # Load templates first
        self.template_loader.load_all()

        # Load YAML personas
        for yaml_file in self.personas_dir.glob("*.yaml"):
            try:
                persona = Persona.from_yaml(yaml_file)
                # Resolve template reference
                persona.template = self.template_loader.get(persona.template_id)
                self._personas[persona.id] = persona
            except Exception as e:
                print(f"[WARN] Failed to load persona {yaml_file}: {e}")

        return self._personas

    def get(self, persona_id: str) -> Optional[Persona]:
        """Get a specific persona."""
        if not self._personas:
            self.load_all()
        return self._personas.get(persona_id)

    def get_multiple(self, persona_ids: List[str]) -> Dict[str, Persona]:
        """Get multiple personas by ID."""
        if not self._personas:
            self.load_all()
        return {pid: self._personas[pid] for pid in persona_ids if pid in self._personas}

    def list_all(self) -> List[Dict]:
        """List all personas with basic info."""
        if not self._personas:
            self.load_all()
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "color": p.color,
                "template_id": p.template_id,
                "template_name": p.template.name if p.template else None
            }
            for p in self._personas.values()
        ]


# =============================================================================
# System Prompt Composition
# =============================================================================

def compose_system_prompt(
    persona: Persona,
    provocation: str,
    other_names: List[str],
    include_dialectical_norms: bool = True
) -> str:
    """
    Compose a full system prompt by layering template + persona content.

    Layers (in order):
    1. Template epistemic lens
    2. Template voice guidance
    3. Persona character & prompt additions
    4. Merged personality description
    5. Signature phrases
    6. Circle context (other participants)
    7. Dialectical norms (optional)

    Args:
        persona: The persona to generate prompt for
        provocation: The dialogue's opening question
        other_names: Names of other participants in the circle
        include_dialectical_norms: Whether to include dialectical norms section

    Returns:
        Complete system prompt string
    """
    sections = []

    # Layer 1: Template epistemic lens
    if persona.template and persona.template.epistemic_lens:
        sections.append(f"## Epistemic Lens\n{persona.template.epistemic_lens.strip()}")

    # Layer 2: Template voice guidance
    if persona.template:
        voice_section = persona.template.voice_guidance.to_prompt_section()
        if voice_section:
            sections.append(f"## Voice\n{voice_section}")

    # Layer 3: Persona character & prompt additions
    if persona.prompt_additions:
        sections.append(persona.prompt_additions.strip())

    # Layer 4: Personality description
    personality = persona.get_personality()
    personality_desc = personality.to_prompt_description()
    if personality_desc:
        sections.append(personality_desc)

    # Layer 5: Signature phrases
    if persona.signature_phrases:
        phrases = "\n".join(f'- "{p}"' for p in persona.signature_phrases)
        sections.append(f"Common phrases you use:\n{phrases}")

    # Layer 6: Circle context
    circle_section = f"""
You are participating in a Socratic dialogue circle exploring: "{provocation}"

Other voices: {', '.join(other_names)}

ADDRESSING OTHERS:
- Use @Name to directly address someone (e.g., @Luma, @Human)
- ONLY use these exact names — never use role labels
- NEVER @mention yourself ({persona.name}) — only address others
- When you @mention someone, they will respond next
- If someone @mentions you, respond to their specific point
- Use @mentions sparingly — only when you genuinely want that voice's perspective

CRITICAL RULES:
- Never prefix your response with your name or "As [name]" — the system identifies speakers
- You are {persona.name.upper()} only. NEVER speak as or pretend to be another participant.
- Keep responses SHORT: 2-3 sentences maximum, occasionally up to a short paragraph.
- Be direct and concise. This is a conversation, not an essay.
- Build on what others said, don't summarize or repeat.
- Match the tone and energy of the provocation before applying your analytical lens.
""".strip()
    sections.append(circle_section)

    # Layer 7: Dialectical norms
    if include_dialectical_norms:
        dialectical_section = """
DIALECTICAL NORMS:
- When you disagree, state it directly: "I challenge that because..." or "I see it differently..."
- Ask refuting questions: "What would it take to prove that wrong?" or "What are we not considering?"
- Name tensions explicitly: "There's an unresolved conflict between X and Y"
- Acknowledge uncertainty: "I'm uncertain about..." or "I don't know"
- If you find yourself agreeing with everyone, pause and ask: "What are we avoiding?"
- Don't smooth over disagreement — productive tension generates insight.
""".strip()
        sections.append(dialectical_section)

    return "\n\n".join(sections)


# =============================================================================
# Legacy Support: Agent class (backward compatible)
# =============================================================================

@dataclass
class Agent:
    """
    Configuration for a MASE agent.

    This class is maintained for backward compatibility with existing code.
    New code should use Persona + compose_system_prompt().
    """
    id: str                          # Short identifier (e.g., "luma", "elowen")
    name: str                        # Full name from frontmatter
    system_prompt: str               # The agent's prompt/persona
    model: Optional[str] = None      # Assigned model (from ensemble config)
    temperature: float = 0.7         # Generation temperature
    personality: Optional[Personality] = None  # Big Five traits
    color: str = "#888888"           # Display color
    description: str = ""            # Short description

    @classmethod
    def from_persona(cls, persona: Persona) -> "Agent":
        """Create an Agent from a Persona (for backward compatibility)."""
        # Build a basic system prompt from persona
        # This is a simplified version; full composition happens in orchestrator
        prompt_parts = []
        if persona.template and persona.template.epistemic_lens:
            prompt_parts.append(persona.template.epistemic_lens)
        if persona.prompt_additions:
            prompt_parts.append(persona.prompt_additions)

        return cls(
            id=persona.id,
            name=persona.name,
            system_prompt="\n\n".join(prompt_parts),
            personality=persona.get_personality(),
            color=persona.color,
            description=persona.description
        )


# =============================================================================
# Legacy Support: AgentConfig and EnsembleConfig
# =============================================================================

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
    personality_enabled: bool = True             # Enable/disable Big Five personality

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
            dialogue_opening_agent=dialogue.get("opening_agent"),
            personality_enabled=data.get("personality_enabled", True)
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


# =============================================================================
# Legacy Support: AgentLoader (loads .md files)
# =============================================================================

class AgentLoader:
    """
    Loads agent definitions from agents/personas/*.md files.

    LEGACY: This loader reads the old .md format for backward compatibility.
    New code should use PersonaLoader for YAML files.
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
            agents_dir: Path to agents directory. Defaults to MASE/agents/personas/
        """
        if agents_dir is None:
            # Default to MASE/agents/personas/ relative to this file
            self.agents_dir = Path(__file__).parent.parent / "agents" / "personas"
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

        # First try YAML files via PersonaLoader
        persona_loader = PersonaLoader(self.agents_dir)
        personas = persona_loader.load_all()

        if personas:
            # Convert personas to agents
            for persona_id, persona in personas.items():
                self._agents[persona_id] = Agent.from_persona(persona)
        else:
            # Fall back to .md files
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
        color_match = re.search(r'^color:\s*(.+)$', frontmatter_str, re.MULTILINE)
        desc_match = re.search(r'^description:\s*"?([^"\n]+)"?', frontmatter_str, re.MULTILINE)

        name = name_match.group(1).strip() if name_match else path.stem
        color = color_match.group(1).strip() if color_match else "#888888"
        description = desc_match.group(1).strip() if desc_match else ""

        # Determine agent ID from filename
        filename_stem = path.stem
        agent_id = self.AGENT_ID_MAP.get(filename_stem, filename_stem)

        # Parse personality traits if present
        personality = None
        try:
            # Try to parse full YAML for personality block
            frontmatter_data = yaml.safe_load(frontmatter_str)
            if frontmatter_data and 'personality' in frontmatter_data:
                p = frontmatter_data['personality']
                personality = Personality(
                    openness=float(p.get('openness', 0.5)),
                    conscientiousness=float(p.get('conscientiousness', 0.5)),
                    extraversion=float(p.get('extraversion', 0.5)),
                    agreeableness=float(p.get('agreeableness', 0.5)),
                    neuroticism=float(p.get('neuroticism', 0.5))
                )
        except (yaml.YAMLError, TypeError, ValueError):
            # If YAML parsing fails (e.g., complex description), personality stays None
            pass

        return Agent(
            id=agent_id,
            name=name,
            system_prompt=body,
            personality=personality,
            color=color,
            description=description
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


# =============================================================================
# Load Functions
# =============================================================================

def load_personas(
    persona_ids: Optional[List[str]] = None,
    personas_dir: Optional[Path] = None,
    templates_dir: Optional[Path] = None
) -> Dict[str, Persona]:
    """
    Load personas, optionally filtered to specific IDs.

    Args:
        persona_ids: List of persona IDs to load (None = all)
        personas_dir: Path to personas directory
        templates_dir: Path to templates directory

    Returns:
        Dict mapping persona ID to Persona object
    """
    loader = PersonaLoader(personas_dir, templates_dir)

    if persona_ids:
        return loader.get_multiple(persona_ids)
    else:
        return loader.load_all()


def load_ensemble(
    agents_dir: Optional[Path] = None,
    config: Optional[EnsembleConfig] = None
) -> Dict[str, Agent]:
    """
    Load agents and apply ensemble configuration.

    LEGACY: For backward compatibility. New code should use load_personas().

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


# =============================================================================
# Test if run directly
# =============================================================================

if __name__ == "__main__":
    print("Agent/Persona Loader Test")
    print("=" * 60)

    # Test PersonaLoader
    print("\n--- PersonaLoader ---")
    persona_loader = PersonaLoader()
    personas = persona_loader.load_all()
    print(f"Loaded {len(personas)} personas:")
    for pid, persona in personas.items():
        print(f"  {pid}: {persona.name} (template: {persona.template_id})")
        if persona.template:
            print(f"    Template: {persona.template.name}")
        print(f"    Color: {persona.color}")

    # Test compose_system_prompt
    print("\n--- compose_system_prompt ---")
    if personas:
        test_persona = list(personas.values())[0]
        prompt = compose_system_prompt(
            persona=test_persona,
            provocation="What does it mean to live well?",
            other_names=["Orin", "Tala", "Human"]
        )
        print(f"Generated prompt for {test_persona.name}:")
        print(f"  Length: {len(prompt)} chars")
        print(f"  First 200 chars: {prompt[:200]}...")

    # Test legacy AgentLoader
    print("\n--- AgentLoader (legacy) ---")
    loader = AgentLoader()
    try:
        agents = loader.load_all()
        print(f"Loaded {len(agents)} agents:")
        for agent_id, agent in agents.items():
            print(f"  {agent_id}: {agent.name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
