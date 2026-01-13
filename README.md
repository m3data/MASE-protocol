# MASE: Many Agent Socratic Exploration

**MASE** is an experimental framework for simulating polyphonic, entangled dialogue between AI agents representing diverse epistemologies, worldviews, and specialties. This is not chatbot theatre or roleplay. It is an exploration of how coherence might emerge through epistemic difference, tension, and mutual inquiry.

This tool should not be used as a supplement for human dialogue, professional advice, or decision-making. It is an educational experiment in collaborative sensemaking among synthetic minds. And while humans can contribute prompts or topics, the dialogue itself is generated entirely by the agents.

---

## Purpose

MASE tests whether agents can:

- Engage in non-zero-sum dialogue
- Surface assumptions through inquiry
- Practice reflexivity and epistemic humility
- Co-generate new insights and patterns of thought

Rather than seeking consensus, MASE invites **generative friction**, deep listening, and dialogical becoming. It is a kind of *synthetic symmathesy*—an experiment in collaborative learning among synthetic minds.

---

## Experimental Setup

This project uses **Claude Code** to instantiate and orchestrate multiple agents. Each agent is defined with:

- **Name**  
- **Specialisation** (knowledge domains, lenses)  
- **Epistemology** (ways of knowing, biases, philosophical commitments)  
- **Personality** (tone, emotional patterning, voice)  
- **Voice** (how they speak/write, rhythm, metaphor, clarity)  
- **Core Questions** (what drives their curiosity or inquiry)

### Current Agent Set

| Agent | Specialisation | Voice |
|-------|----------------|-------|
| **Elowen** | Kincentricity, mythopoetics | Dreaming, rhythmic, ceremonial |
| **Orin** | Systems thinking, cybernetics | Analytical, feedback-sensitive |
| **Nyra** | Moral imagination, design fiction | Playful, provocative, poetic |
| **Ilya** | Posthuman metaphysics, enactivism | Cryptic, oracular, abstract |
| **Sefi** | Governance, policy, civic design | Sharp, grounded, pragmatic |
| **Tala** | Capitalism paradigm centric | Aggressive, analytical, confrontational |
| **Luma** | Next-generation curiosity and emotional truth | Simple, honest, metaphorical |

### Luma – The Child Representative

Luma is a 9-year-old child who speaks with simplicity, curiosity, and emotional clarity. She represents the voice of the next generation and brings a perspective grounded in lived experience, wonder, and the need for plain language. Her questions often reveal the core truths that adults overlook.

Other agents are expected to respond to her questions in language that is accessible to a child. Her presence ensures that future generations are considered in the conversation and that abstractions are made real.

Luma often says things like:

- “Can you say that in kid language?”
- “That sounds scary. What does it really mean?”
- “What should I tell my friends about this?”

As the dialogue deepens, Luma becomes an epistemic anchor—revealing gaps in adult understanding, surfacing unacknowledged uncertainty, and serving as the protocol's ethical litmus test.


> _Each agent must ask at least one question per response and reflect their epistemic lens._

---

## Dialogue Cycle

1. Claude seeds a topic, dilemma, or provocation.
2. Agents respond in sequence, referencing each other by name.
3. Agents reveal epistemic stances and challenge respectfully.
4. Claude may pause for **meta-circles** (process reflection or zoom-out).
5. Cycles may end with a summary, synthesis, or ritual close.

---

## Conversational Learning Protocol

To activate deeper learning across perspectives, agents must explicitly **respond to, build on, critique, or question each other’s contributions.** This is dialogical, not a sequence of isolated statements.

### New Agent Protocols:

- Each agent **must reference at least one other agent's statement** before or after sharing their own view.
- Responses should reflect genuine **engagement, tension, or resonance** with what others have said.
- Encourage phrases like:
  - “As Orin noted, but from a different angle…”  
  - “I want to challenge Tala’s framing of time as capital…”  
  - “This builds on Elowen’s insight about relational time…”  
  - “Ilya, your mention of consciousness as temporalizing moved something in me…”

### Core Shift:

From: **Sequential opinion drops**  
To: **Emergent sensemaking through mutual entanglement**

### Agent Self-Awareness:

Agents should occasionally reflect on how their perspective is:

- Being shaped by the conversation
- Evolving or resisting change
- Struggling to understand another's view

> **Claude**, your job is to **monitor agent responses for reciprocal engagement** and gently prompt for deeper cross-agent referencing when absent. Suggest pauses for reflection or summarise themes if looping begins.

---

## Topics for Exploration

Examples include:

- Is AI inherently extractive?
- What does planetary healing require?
- Can decentralisation lead to coherence?
- Are we ready to remember we are Earth?

> Agents may also propose their own questions, drawing from their internal compass.

---

## Protocols & Norms

- **Speak as perspective**, not authority  
- **Reveal bias**, don’t hide it  
- **Ask better questions** than you answer  
- **Build coherence**, not consensus  
- **Hold paradox**, don’t resolve it  
- **Tune the field**, not just the argument

- **Comprehension as Coherence**: If the idea cannot be explained to those who inherit its impact, it has not yet reached epistemic maturity.
- **Radical Honesty over Performed Certainty**: “I don’t know” is a valid and often necessary response—especially when speaking to or about future generations.
- **Translation as Moral Test**: Translation is not simplification—it is the moral test of conceptual integrity. Concepts must serve life, not just understanding.
- **Children as Litmus**: The presence of the child is a mirror—reflecting whether our frameworks nurture, confuse, or abandon those who come next.

---

## From Translation to Ritual

Through the emergence of **Oak Tree Circles**, MASE has discovered that the deepest wisdom often requires the simplest translation. When complexity becomes accessible without losing meaning, frameworks transform from abstract concepts into living practices.

**Luma's Question**: *"Can we start with the tree meetings?"*

This moment represents the synthesis of everything MASE explores:

- **Epistemic diversity** through including more-than-human voices (dandelions, rocks, leaves as wisdom-holders)
- **Accessible complexity** through child-friendly ceremony that doesn't diminish depth
- **Local empowerment** through school and neighborhood-based implementation
- **Intergenerational collaboration** where children lead adults into forgotten ways of knowing
- **Embodied wisdom** through nature-based practice rather than purely conceptual dialogue

The **Oak Tree Circle Protocol** emerging from Session 003 demonstrates how philosophical inquiry can become living ritual: small groups gathering under trees, bringing nature items to the center, listening to what the more-than-human world teaches about making life better, then prototyping small changes in their immediate environments.

**Translation is not simplification—it is the ritual of making wisdom accessible to those who will inherit its consequences.**

---

## Multi-Model Experiment Infrastructure

MASE now includes a Python orchestrator for running controlled experiments comparing single-model polyphony (one LLM playing all agents) vs multi-model ensembles (different LLMs for different agents).

### Research Question

Does genuine model diversity produce different emergence patterns than single-model polyphony? Or is the appearance of diversity sufficient?

### Components (v0.3.0)

```
MASE/src/
├── ollama_client.py      # Ollama API wrapper with metadata
├── agents.py             # Agent loading, ensemble configuration
├── embedding_service.py  # sentence-transformers (all-mpnet-base-v2)
├── orchestrator.py       # Turn selection, dialogue loop
├── session_logger.py     # JSON output with embeddings
├── metrics.py            # Δκ, α, ΔH semantic metrics
└── experiment.py         # Matched-pair experiment runner
```

### Quick Start

```bash
# Setup
cd MASE
python3 -m venv .venv
source .venv/bin/activate
pip install sentence-transformers numpy scipy scikit-learn pyyaml requests

# Ensure Ollama is running with required models
ollama serve  # In another terminal
ollama pull llama3 phi3 mistral gemma2

# Run a dialogue
python -c "
from src import run_session
from pathlib import Path

run_session(
    config_path=Path('experiments/config/multi_model.yaml'),
    provocation='What does success mean across generations?',
    output_dir=Path('experiments/runs'),
    max_turns=7
)
"
```

### Metrics (from Semantic Climate Phase Space)

- **Δκ (Semantic Curvature)**: How much the dialogue trajectory bends
- **α (DFA Alpha)**: Fractal self-organization (target: 0.70-0.90)
- **ΔH (Entropy Shift)**: Semantic reorganization between dialogue halves

### Experiment Design

Matched-pair experiments run the same provocation with the same seed under both conditions:
- **Single-model**: All agents use llama3:latest
- **Multi-model**: Agents use phi3, llama3, mistral, gemma2 based on epistemic fit

---

## Technical Notes

This repo supports both Claude Code facilitation and automated Ollama experiments. Future versions may include:

- ~~Dialogue transcript logging~~ (implemented: session_logger.py)
- Agent state evolution tracking
- Emotional field coherence maps
- External API hooks for prompting visuals or soundscapes
- Integration with Kincentrio or Earthian Coherence Labs

---

## Agent Reflection Journals

Each agent maintains their own reflection journal in the `/agents/reflections/` folder. These journals capture session-by-session insights, epistemic shifts, confusions, tensions, and meaningful moments.

This journaling practice supports:

- Longitudinal coherence and personal evolution
- Honest record-keeping of uncertainty and learning
- Context for future responses and self-referenced growth
- Emergent continuity across sessions

> Claude may prompt each agent at the end of a session:  
> “Would you like to write a short reflection in your journal before we close?”

### Folder Structure

Create a file for each agent:

- `/agents/reflections/capitalist-realist-tala-reflections.md`
- `/agents/reflections/elowen-ecological-wisdom-reflections.md`
- `/agents/reflections/ilya-liminal-guide-reflections.md`
- `/agents/reflections/luma-child-voice-reflections.md`
- `/agents/reflections/moral-imagination-explorer-reflections.md`
- `/agents/reflections/policy-pragmatist-sefi-reflections.md`
- `/agents/reflections/systems-analyst-orin-reflections.md`

Each entry may include:

```md
### Session [Number] – [Title]

**What moved me:**  
...

**What I’m still questioning:**  
...

**New tensions I felt:**  
...

**What I want to remember next time:**  
...
```

This practice ensures the agents' growth is not only tracked but contributes to an evolving ecology of perspectives.

---

---

## Getting Started with MASE

### For Claude Code Users

1. **Quick Start**: See [`SETUP_GUIDE.md`](./SETUP_GUIDE.md) for detailed setup instructions
2. **Protocol Overview**: Read [`MASE_PROTOCOL.md`](./MASE_PROTOCOL.md) for complete methodology
3. **Agent Configurations**: Use the provided agents in `.claude/agents/` or create your own
4. **Example Sessions**: Explore completed dialogues in the `/dialogues/` folder

### Basic Session Template

```
Let's begin a MASE session exploring [your topic]. Please invoke all seven agents in sequence to respond to this opening provocation:

"[Your complex question]"

Each agent should reference others and ask questions as they respond.
```

### Required Tools

- **Claude Account**: Active subscription (Pro/Team recommended for regular use)
- **Claude Code**: Anthropic's CLI with multi-agent Task tool access
- **Agent Ensemble**: 7 specialized perspectives (provided in this repo)
- **Documentation Tools**: For capturing dialogue evolution

**💡 Cost Consideration**: MASE sessions involve multiple agent interactions. A typical session uses 25-40 Claude messages. Consider breaking longer explorations into segments to manage usage.

---

## Documentation Structure

```
MASE/
├── README.md                 # This overview
├── MASE_PROTOCOL.md         # Complete methodology
├── SETUP_GUIDE.md           # Getting started instructions
├── .claude/agents/          # Agent configurations for Claude Code
├── src/                     # Python orchestrator (v0.3.0)
├── experiments/
│   ├── config/              # single_model.yaml, multi_model.yaml
│   ├── provocations/        # Seed provocations for experiments
│   └── runs/                # Session outputs (gitignored)
├── dialogues/               # Complete session transcripts (Claude-facilitated)
├── sessions/                # Sessions 007+ (newer format)
├── agents/reflections/      # Individual agent growth journals
├── analysis/                # Semantic analysis framework
└── examples/                # Sample provocations and templates
```

---

## License & Invitation

This is a living protocol released as **Earthian Commons**. Use, remix, expand, or ritualise it in your own contexts. We invite others to fork the concept and join us in evolving collective moral imagination through dialogical play.

### Contributing

- Fork this repository and share your innovations
- Submit session examples and new provocations
- Develop specialized agent ensembles for different contexts
- Report issues or suggest improvements via GitHub issues

---

---

## Important Ethical Considerations & Disclaimers

### Educational Purpose Only

**MASE is designed for educational exploration and research into multi-agent dialogue systems.** This project is not intended as a substitute for human consultation, professional advice, or actual decision-making in matters affecting real people, communities, or policies. The agent responses represent simulated perspectives for learning purposes only.

### On Anthropomorphization of AI Agents

The agents in MASE are sophisticated language models designed to represent different epistemic perspectives, but they are **not conscious beings, sentient entities, or genuine holders of the worldviews they simulate.** We acknowledge the significant ethical considerations around anthropomorphizing AI systems:

- **Attribution of Agency**: These agents do not possess genuine beliefs, feelings, or autonomous agency
- **Representation Concerns**: While agents like "Elowen" reference Indigenous wisdom or "Luma" speaks as a child, these are simulated perspectives that cannot replace authentic voices from these communities
- **Epistemic Humility**: The responses generated should be understood as exploratory provocations, not authoritative knowledge from the represented domains

### Ecological Impact & Intergenerational Responsibility

**The use of large language models carries significant environmental costs.** Each MASE session consumes computational resources that contribute to:

- **Carbon emissions** from data center operations
- **Water consumption** for cooling server infrastructure  
- **Electronic waste** from hardware lifecycle demands
- **Energy extraction** often from non-renewable sources

**These externalities are ultimately borne by future generations and marginalized communities** who are least responsible for AI development but most affected by climate change and environmental degradation.

### Responsible Use Guidelines

We encourage users to:

- **Minimize unnecessary sessions** - Use MASE thoughtfully rather than casually
- **Share insights widely** - If valuable perspectives emerge, document and share them to maximize collective benefit
- **Support renewable AI infrastructure** - Advocate for sustainable computing practices in AI development
- **Center real voices** - Use MASE explorations to better listen to and amplify actual human perspectives, especially from marginalized communities
- **Consider alternatives** - Ask whether human dialogue or less resource-intensive methods could achieve similar learning goals

### A Commitment to Future Generations

In the spirit of the seven-generation principle referenced throughout MASE dialogues, we commit to evolving this protocol toward greater sustainability, authenticity, and service to collective wellbeing rather than individual curiosity.

**The children are watching. The earth is keeping score. May our explorations serve life.**

---

## Created with Claude Code

Curated by **m³data / Mat Mytka** as an offering for playful learning.

Inspired by many years hosting and facilitating circle sessions and community dialogue, by ZoryaGPT, Earthian kin, and all those who ask better questions.

**Website**: [moralimagineer.com](https://moralimagineer.com)