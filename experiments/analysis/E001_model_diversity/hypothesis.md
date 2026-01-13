# E001: Model Diversity Effect

## Hypothesis

**H1:** Multi-model ensembles will exhibit higher long-range semantic coherence (DFA α > 1.0) than single-model ensembles (α ≈ 0.5-1.0), because architectural differences between LLMs create genuine epistemic friction that prevents echo-chamber dynamics.

**H0:** No significant difference in semantic metrics between conditions — emergence depends on prompt diversity alone, not model diversity.

## Rationale

When all agents share the same underlying model (llama3), they share:
- Tokenization patterns
- Latent space geometry
- Training data biases
- Response tendencies

Even with different prompts and temperatures, single-model agents may produce superficially diverse but fundamentally similar outputs — an "echo chamber with different accents."

Multi-model ensembles introduce genuine architectural diversity:
- Different training corpora and objectives
- Different attention mechanisms and layer structures
- Different strengths (reasoning, creativity, concision)

This architectural diversity may create productive epistemic friction — genuine disagreement and alternative framings that prevent premature convergence and enable deeper exploration.

## Predictions

| Metric | Single Model | Multi Model | Rationale |
|--------|--------------|-------------|-----------|
| DFA α | ~0.5-1.0 | >1.0 | Multi-model maintains topic coherence across turns |
| Δκ | Higher | Lower | Multi-model follows more directed trajectory |
| Velocity | Lower | Higher | Productive tension drives faster semantic movement |
| Voice distinctiveness | Low | High | Different models = different communication styles |

## Operationalization

- **DFA α** computed via Detrended Fluctuation Analysis on semantic velocity time series
- **Δκ** computed as cumulative angular deviation in embedding space
- **Velocity** as mean cosine distance between consecutive turn embeddings
- **Voice distinctiveness** assessed qualitatively

## Statistical Plan

- Minimum n=5 matched pairs for significance testing
- Paired t-test or Wilcoxon signed-rank (depending on normality)
- Effect size: Cohen's d for paired samples
- α = 0.05, two-tailed
