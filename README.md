# MASE: Many Agent Socratic Exploration

![Repo Status](https://img.shields.io/badge/REPO_STATUS-Active_Research-blue?style=for-the-badge&labelColor=8b5e3c&color=e5dac1)
![Version](https://img.shields.io/badge/VERSION-0.7.0-blue?style=for-the-badge&labelColor=3b82f6&color=1e40af)
![License](https://img.shields.io/badge/LICENSE-ESL--A-green?style=for-the-badge&labelColor=10b981&color=047857)

An experimental framework for **polyphonic dialogue** between AI agents representing diverse epistemologies. Not chatbot theatre—an exploration of how coherence emerges through epistemic difference, tension, and mutual inquiry.

---

## What MASE Does

Seven AI agents engage in structured dialogue around provocations you provide. Each agent brings a distinct epistemology:

| Agent | Lens | Voice |
|-------|------|-------|
| **Elowen** | Ecological wisdom, kincentricity | Ceremonial, rhythmic |
| **Orin** | Systems thinking, cybernetics | Analytical, structural |
| **Nyra** | Moral imagination, design fiction | Playful, provocative |
| **Ilya** | Posthuman metaphysics, liminal | Cryptic, paradox-holding |
| **Sefi** | Governance, policy, civic design | Sharp, pragmatic |
| **Tala** | Capitalism, markets, power | Challenging, ROI-focused |
| **Luma** | Child voice (9 years old) | Simple, honest, devastating |

**Luma as epistemic anchor**: All abstractions must be translatable to child-accessible language. If you can't explain it to Luma, you haven't understood it.

---

## Quick Start

### Interactive Web App (v0.7.0)

```bash
# Setup
cd MASE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ensure Ollama is running
ollama serve  # In another terminal
ollama pull llama3 phi3 mistral gemma2:9b

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

---

## Research Context

MASE is part of the [EarthianLabs](https://github.com/EarthianLabs) research ecosystem investigating **transformative adaptation**—how individuals and collectives develop adaptive capacity under systemic stress.

### Research Questions

- How does epistemic diversity affect dialogue coherence?
- What conditions produce genuine inquiry vs performative mimicry?
- Can semantic metrics detect emergence and stuck patterns?

### Completed Experiments

- **E001**: Model diversity effect (multi-model vs single-model)
- **E002**: Personality system effect (in progress)

See `experiments/` for protocol, configs, and results.

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

## Project Structure

```
MASE/
├── src/                    # Python backend (v0.7.0)
│   ├── server.py           # Flask API + SSE streaming
│   ├── orchestrator.py     # Dialogue loop, turn selection
│   ├── session_analysis.py # Post-hoc semantic analysis
│   ├── basins.py           # Basin detection from SC
│   └── metrics.py          # Δκ, α, ΔH computation
├── web/                    # Frontend
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── .claude/agents/         # Agent definitions
├── experiments/            # Research pipeline
├── sessions/               # Saved dialogues (gitignored)
└── dialogues/              # Historic sessions 001-006
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

## Links

- **Detailed Protocol**: [MASE_PROTOCOL.md](./MASE_PROTOCOL.md)
- **Setup Guide**: [SETUP_GUIDE.md](./SETUP_GUIDE.md)
- **Contributing**: [CONTRIBUTING.md](./CONTRIBUTING.md)

---

*"Maybe we don't need a big plan. Maybe we need a lot of small true things that people can teach each other."* — Luma, Session 008
