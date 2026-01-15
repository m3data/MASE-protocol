# MASE: Many Agent Socratic Exploration

![Repo Status](https://img.shields.io/badge/REPO_STATUS-Active_Research-blue?style=for-the-badge&labelColor=8b5e3c&color=e5dac1)
![Version](https://img.shields.io/badge/VERSION-0.7.0-blue?style=for-the-badge&labelColor=3b82f6&color=1e40af)
![License](https://img.shields.io/badge/LICENSE-ESL--A-green?style=for-the-badge&labelColor=10b981&color=047857)

An experimental framework for **polyphonic dialogue** between AI agents representing diverse epistemologies. Not chatbot theatre—an exploration of how coherence emerges through epistemic difference, tension, and mutual inquiry.

---

## What MASE Does

Seven AI agents engage in structured dialogue around provocations you provide. Each agent brings a distinct epistemology and voice:

| Agent | Lens | Model | Voice |
|-------|------|-------|-------|
| **Elowen** | Ecological wisdom, kincentricity | llama3 | Ceremonial, rhythmic |
| **Orin** | Systems thinking, cybernetics | mistral | Analytical, structural |
| **Nyra** | Moral imagination, design fiction | gemma2 | Playful, provocative |
| **Ilya** | Posthuman metaphysics, liminal | llama3 | Cryptic, paradox-holding |
| **Sefi** | Governance, policy, civic design | mistral | Sharp, pragmatic |
| **Tala** | Capitalism, markets, power | gemma2 | Challenging, ROI-focused |
| **Luma** | Child voice (9 years old) | llama3.2 | Simple, honest, devastating |

**Luma as epistemic anchor**: All abstractions must be translatable to child-accessible language. If you can't explain it to Luma, you haven't understood it.

---

## Quick Start

```bash
# Setup
cd MASE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ensure Ollama is running with required models
ollama serve  # In another terminal
ollama pull llama3 llama3.2 mistral gemma2:9b

# Start the server
python src/server.py
```

Open **http://localhost:5050** and enter a provocation to begin.

### Features

- **Human participation**: Join the circle as the 8th voice
- **Real-time streaming**: Watch the dialogue unfold via SSE
- **Post-session analysis**: End & Analyze triggers semantic analysis
  - Basin detection (Collaborative Inquiry, Cognitive Mimicry, etc.)
  - Coherence pattern classification
  - DFA alpha, semantic curvature, voice distinctiveness
- **Research pipeline**: Analysis saved for experimental comparison
- **Resume capability**: Recover interrupted experiments from checkpoint

---

## Personality System

Each agent has a Big Five (OCEAN) personality profile that influences their sampling parameters:

| Trait | High → | Low → |
|-------|--------|-------|
| **Openness** | Higher temperature | Lower temperature |
| **Conscientiousness** | Lower top_p | Higher top_p |
| **Extraversion** | More tokens | Fewer tokens |
| **Agreeableness** | Lower temperature | Higher temperature |
| **Neuroticism** | Higher variability | More stable |

Personalities are defined in `.claude/agents/*.md` files with YAML frontmatter.

---

## Dialogue Norms

Agents follow these conversational protocols:

- **Reference others**: Each agent must engage with what others have said
- **Ask questions**: Every response includes at least one genuine question
- **Reveal bias**: State epistemic stance openly, don't hide it
- **Build coherence, not consensus**: Productive tension over premature agreement
- **Hold paradox**: Don't resolve what needs to remain open

---

## Semantic Metrics

Borrowed from [Semantic Climate Phase Space](../semantic-climate-phase-space/):

| Metric | What it measures |
|--------|------------------|
| **DFA α** | Long-range correlation (0.5=noise, 1.0=pink noise) |
| **Δκ** | Semantic curvature—trajectory complexity |
| **ΔH** | Entropy shift—semantic reorganization |
| **Ψ** | Composite coupling vector across substrates |

### Basin Detection

Dialogues are classified into attractor basins:

- **Collaborative Inquiry** — Genuine exploration, productive tension
- **Cognitive Mimicry** — Performing engagement without uncertainty
- **Deep Resonance** — Aligned meaning-making
- **Generative Conflict** — Productive disagreement
- **Sycophantic Convergence** — Premature agreement

---

## Research Context

MASE is part of the [EarthianLabs](https://github.com/m3data) research ecosystem investigating **transformative adaptation**—how individuals and collectives develop adaptive capacity under systemic stress.

### Research Questions

- How does epistemic diversity affect dialogue coherence?
- What conditions produce genuine inquiry vs performative mimicry?
- Can semantic metrics detect emergence and stuck patterns?

### Experiments

| ID | Hypothesis | Status |
|----|------------|--------|
| **E001** | Multi-model ensembles produce higher DFA α than single-model | Complete (not supported) |
| **E002** | Personality system increases inquiry ratio | In progress |

See `experiments/PROTOCOL.md` for methodology.

---

## Project Structure

```
MASE/
├── src/
│   ├── server.py              # Flask API + SSE streaming
│   ├── orchestrator.py        # Dialogue loop, turn selection
│   ├── interactive_orchestrator.py  # Web mode with human participation
│   ├── session_analysis.py    # Post-hoc semantic analysis
│   ├── basins.py              # Basin detection
│   ├── metrics.py             # Δκ, α, ΔH computation
│   ├── agents.py              # Agent loading, personality system
│   ├── embedding_service.py   # sentence-transformers embeddings
│   ├── ollama_client.py       # Ollama API wrapper
│   └── experiment.py          # Matched-pair experiment runner
├── web/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── .claude/agents/            # Agent definitions with personalities
├── experiments/
│   ├── PROTOCOL.md            # Experiment methodology
│   ├── config/                # Model configurations
│   └── runs/                  # Session data (gitignored)
├── sessions/                  # Interactive session checkpoints
└── dialogues/                 # Historic sessions 001-008
```

---

## Ethics & Considerations

### These agents are not conscious

They simulate perspectives—they don't hold beliefs. Elowen references Indigenous wisdom but cannot replace Indigenous voices. Luma speaks as a child but is not a child.

### Environmental cost

Each session consumes compute resources with real environmental impact. Use thoughtfully. Share insights widely.

### Not a substitute for human dialogue

MASE is for research and learning. It cannot replace genuine human conversation, professional advice, or community decision-making.

---

## License

[Earthian Stewardship License (ESL-A)](./LICENSE)

- Respect somatic sovereignty
- No manipulation or surveillance
- Non-commercial by default
- Share safety improvements

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on:
- Forking and sharing innovations
- Submitting session examples
- Developing specialized agent ensembles
- Reporting issues

---

*"Maybe we don't need a big plan. Maybe we need a lot of small true things that people can teach each other."* — Luma, Session 008
