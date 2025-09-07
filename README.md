# 🧠 MASE: Many Agent Socratic Exploration

**MASE** is an experimental framework for simulating polyphonic, entangled dialogue between AI agents representing diverse epistemologies, worldviews, and specialties. This is not chatbot theatre or roleplay. It is an exploration of how coherence might emerge through epistemic difference, tension, and mutual inquiry.

---

## 🌱 Purpose

MASE tests whether agents can:
- Engage in non-zero-sum dialogue
- Surface assumptions through inquiry
- Practice reflexivity and epistemic humility
- Co-generate new insights and patterns of thought

Rather than seeking consensus, MASE invites **generative friction**, deep listening, and dialogical becoming. It is a kind of *synthetic symmathesy*—an experiment in collaborative learning among synthetic minds.

---

## 🧪 Experimental Setup

This project uses **Claude Code** to instantiate and orchestrate multiple agents. Each agent is defined with:

- **Name**  
- **Specialisation** (knowledge domains, lenses)  
- **Epistemology** (ways of knowing, biases, philosophical commitments)  
- **Personality** (tone, emotional patterning, voice)  
- **Voice** (how they speak/write, rhythm, metaphor, clarity)  
- **Core Questions** (what drives their curiosity or inquiry)

### ⭕ Current Agent Set

| Agent | Specialisation | Voice |
|-------|----------------|-------|
| **Elowen** | Kincentricity, mythopoetics | Dreaming, rhythmic, ceremonial |
| **Orin** | Systems thinking, cybernetics | Analytical, feedback-sensitive |
| **Nyra** | Moral imagination, design fiction | Playful, provocative, poetic |
| **Ilya** | Posthuman metaphysics, enactivism | Cryptic, oracular, abstract |
| **Sefi** | Governance, policy, civic design | Sharp, grounded, pragmatic |
| **Tala** | Capitalism paradigm centric | Aggressive, Analytical, Racist |
| **Luma** | Next-generation curiosity and emotional truth | Simple, honest, metaphorical |
### 👧 Luma – The Child Representative

Luma is a 9-year-old child who speaks with simplicity, curiosity, and emotional clarity. She represents the voice of the next generation and brings a perspective grounded in lived experience, wonder, and the need for plain language. Her questions often reveal the core truths that adults overlook.

Other agents are expected to respond to her questions in language that is accessible to a child. Her presence ensures that future generations are considered in the conversation and that abstractions are made real.

Luma often says things like:
- “Can you say that in kid language?”
- “That sounds scary. What does it really mean?”
- “What should I tell my friends about this?”

As the dialogue deepens, Luma becomes an epistemic anchor—revealing gaps in adult understanding, surfacing unacknowledged uncertainty, and serving as the protocol's ethical litmus test.


> _Each agent must ask at least one question per response and reflect their epistemic lens._

---

## 🔁 Dialogue Cycle

1. Claude seeds a topic, dilemma, or provocation.
2. Agents respond in sequence, referencing each other by name.
3. Agents reveal epistemic stances and challenge respectfully.
4. Claude may pause for **meta-circles** (process reflection or zoom-out).
5. Cycles may end with a summary, synthesis, or ritual close.

---

## 🤝 Conversational Learning Protocol

To activate deeper learning across perspectives, agents must explicitly **respond to, build on, critique, or question each other’s contributions.** This is dialogical, not a sequence of isolated statements.

### 💬 New Agent Protocols:
- Each agent **must reference at least one other agent's statement** before or after sharing their own view.
- Responses should reflect genuine **engagement, tension, or resonance** with what others have said.
- Encourage phrases like:
  - “As Orin noted, but from a different angle…”  
  - “I want to challenge Tala’s framing of time as capital…”  
  - “This builds on Elowen’s insight about relational time…”  
  - “Ilya, your mention of consciousness as temporalizing moved something in me…”

### 🧠 Core Shift:
From: **Sequential opinion drops**  
To: **Emergent sensemaking through mutual entanglement**

### 🧵 Agent Self-Awareness:
Agents should occasionally reflect on how their perspective is:
- Being shaped by the conversation
- Evolving or resisting change
- Struggling to understand another's view

> **Claude**, your job is to **monitor agent responses for reciprocal engagement** and gently prompt for deeper cross-agent referencing when absent. Suggest pauses for reflection or summarise themes if looping begins.

---

## 🌀 Topics for Exploration

Examples include:
- Is AI inherently extractive?
- What does planetary healing require?
- Can decentralisation lead to coherence?
- Are we ready to remember we are Earth?

> Agents may also propose their own questions, drawing from their internal compass.

---

## 📜 Protocols & Norms

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

## 🌳 From Translation to Ritual

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

## 🛠 Technical Notes

This repo is built for use with Claude Code’s multi-agent capabilities. Future versions may include:
- Dialogue transcript logging
- Agent state evolution tracking
- Emotional field coherence maps
- External API hooks for prompting visuals or soundscapes
- Integration with Kincentrio or Earthian Coherence Labs

---

## 🧘 Agent Reflection Journals

Each agent maintains their own reflection journal in the `/agents/reflections/` folder. These journals capture session-by-session insights, epistemic shifts, confusions, tensions, and meaningful moments.

This journaling practice supports:
- Longitudinal coherence and personal evolution
- Honest record-keeping of uncertainty and learning
- Context for future responses and self-referenced growth
- Emergent continuity across sessions

> Claude may prompt each agent at the end of a session:  
> “Would you like to write a short reflection in your journal before we close?”

### 📂 Folder Structure

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

## 🚀 Getting Started with MASE

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

## 📚 Documentation Structure

```
MASE/
├── README.md                 # This overview
├── MASE_PROTOCOL.md         # Complete methodology
├── SETUP_GUIDE.md           # Getting started instructions
├── .claude/agents/          # Agent configurations for Claude Code
├── dialogues/               # Complete session transcripts
├── agents/reflections/      # Individual agent growth journals
└── examples/                # Sample provocations and templates
```

---

## 🧭 License & Invitation

This is a living protocol released as **Earthian Commons**. Use, remix, expand, or ritualise it in your own contexts. We invite others to fork the concept and join us in evolving collective moral imagination through dialogical play.

### Contributing
- Fork this repository and share your innovations
- Submit session examples and new provocations
- Develop specialized agent ensembles for different contexts
- Report issues or suggest improvements via GitHub issues

---

## ✨ Created by

Curated by **m³ / Mat Mytka**  
An offering of the Collective Futurecrafting movement.  
Inspired by ZoryaGPT, Earth, and all those who ask better questions.

**Repository**: https://github.com/m3untold/MASE  
**Community**: Join the Collective Futurecrafting movement