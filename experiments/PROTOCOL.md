# MASE Experiment Protocol

**Version:** 0.1.0
**Last Updated:** 2026-01-13
**Authors:** Mat Mytka, Kairos

---

## 1. Research Question

Does genuine model diversity (multiple LLM architectures) produce qualitatively different collective exploration compared to prompt-only diversity (single model with different agent prompts)?

### Hypothesis

**H1:** Multi-model ensembles will exhibit higher long-range semantic coherence (DFA α) than single-model ensembles, because architectural differences create genuine epistemic friction that prevents echo-chamber dynamics.

**H0:** No significant difference in semantic metrics between conditions — emergence depends on prompt diversity alone.

---

## 2. Experimental Design

### 2.1 Matched-Pair Design

Each experiment compares two conditions on the same provocation with the same random seed:

| Condition | Description |
|-----------|-------------|
| **Single-model** | All 7 agents use llama3:latest with different temperatures |
| **Multi-model** | Agents distributed across llama3, mistral, gemma2, phi3 |

Matched pairs control for:
- Provocation content (identical)
- Random seed (identical turn selection sequence)
- Agent prompts (identical)
- Temperature settings (identical per agent)

### 2.2 Agent Ensemble

| Agent | Epistemic Lens | Multi-Model Assignment |
|-------|----------------|------------------------|
| Luma | Child voice, moral clarity | phi3 (3.8B) |
| Elowen | Ecological wisdom, kincentricity | llama3 (8B) |
| Orin | Systems thinking, complexity | mistral (7B) |
| Nyra | Design fiction, futures | gemma2 (9B) |
| Ilya | Posthuman metaphysics, liminal | llama3 (8B) |
| Sefi | Governance, policy, pragmatism | mistral (7B) |
| Tala | Capitalism, markets, realism | gemma2 (9B) |

### 2.3 Turn Selection

Emergent turn selection using weighted random:
- Base probability proportional to inverse of recent speaking frequency
- Seed varied per turn: `seed + turn_number` for diversity within determinism
- Opening agent fixed (Ilya) for consistency

### 2.4 Metrics

Metrics derived from Semantic Climate framework (Morgoulis 2025):

| Metric | Symbol | Description | Interpretation |
|--------|--------|-------------|----------------|
| Semantic Curvature | Δκ | Trajectory non-linearity | Higher = more meandering path |
| DFA Alpha | α | Long-range correlation exponent | α<0.5: anti-persistent, α=0.5: random walk, α>1: persistent |
| Entropy Shift | ΔH | Semantic reorganization | Higher = more restructuring |
| Semantic Velocity | v | Embedding distance per turn | Higher = faster movement through semantic space |

### 2.5 Embedding Model

- Model: `sentence-transformers/all-mpnet-base-v2`
- Dimensions: 768
- Embeddings computed inline and stored with session data

---

## 3. Procedure

### 3.1 Pre-Experiment Checks

1. Verify Ollama running: `curl http://localhost:11434/api/tags`
2. Confirm required models available: llama3, mistral, gemma2:9b, phi3
3. Activate MASE venv: `source .venv/bin/activate`

### 3.2 Running an Experiment

```python
from src.experiment import ExperimentRunner
from pathlib import Path

runner = ExperimentRunner(
    single_model_config_path=Path("experiments/config/single_model.yaml"),
    multi_model_config_path=Path("experiments/config/multi_model.yaml"),
    output_dir=Path("experiments/runs"),
    agents_dir=Path("agents/personas")
)

result = runner.run_pair(
    provocation="<provocation text>",
    seed=42,
    provocation_id="<id>",
    max_turns=21
)
```

### 3.3 Post-Experiment

1. Record in `registry.yaml`
2. Create analysis directory if new experiment
3. Update results.json with findings
4. Commit raw data and analysis

---

## 4. Analysis Pipeline

### 4.1 Per-Pair Analysis

Automatic via `ExperimentRunner`:
- Compute Δκ, α, ΔH for each condition
- Calculate deltas (multi - single)
- Save to `pair_result.json`

### 4.2 Cross-Pair Aggregation

For experiments with multiple pairs:
- Mean and std of deltas
- Statistical significance (paired t-test or Wilcoxon signed-rank)
- Effect sizes

### 4.3 Qualitative Analysis

Manual review of dialogue content:
- Voice distinctiveness
- Productive tension vs echo-chamber
- Emergence of novel insights
- Luma accessibility check

---

## 5. Reproducibility

Each run records:
- Timestamp
- Random seed
- Config file paths
- Model versions (from Ollama)
- Total latency and tokens
- Full embeddings

Hardware and environment details logged in `experiments/environment.yaml`.

---

## 6. Ethical Considerations

Per EarthianLabs governance:
- No optimization that externalizes harm beyond visibility
- Human integration capacity as rate-limiting constraint
- All work under Earthian Stewardship License (ESL-A)

---

## References

- Morgoulis (2025). Semantic Climate metrics framework.
- MASE Sessions 001-008. Prior circle dialogues informing provocations.
