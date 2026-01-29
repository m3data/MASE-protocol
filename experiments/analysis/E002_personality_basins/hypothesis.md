# E002: Personality System Effect on Basin Distribution

**Status:** Designed
**Date:** 2026-01-13
**Depends on:** E001 (model diversity), v0.6.0 (Big Five personality system)

---

## Background

E001 tested model diversity using DFA α and found H1 not supported (p=0.43). However, re-analysis with basin detection reveals:

| Metric | Multi-Model Delta | Pairs Favoring |
|--------|------------------|----------------|
| Inquiry Ratio | +0.24 | 4/7 |
| Voice Distinctiveness | +0.08 | 6/7 |
| Locked Patterns | -4.9 | 3/7 |

**Key insight:** Effect is provocation-dependent. Systems/economic framing benefits from diversity; relational framing does not.

v0.6.0 introduced Big Five personality traits that map to:
- **Prompt descriptions** (extraversion, agreeableness → voice)
- **Sampling parameters** (openness → temperature, conscientiousness → top_p)

---

## Research Question

Does the Big Five personality system shift basin distribution toward more genuine exploration?

---

## Hypotheses

### H1 (Primary): Personality Diversity Increases Collaborative Inquiry

**Prediction:** Sessions with personality-enabled agents will spend more time in "Collaborative Inquiry" and less in "Cognitive Mimicry" compared to base-temperature sessions.

**Rationale:** Personality traits create behavioral diversity through both prompt and parameter space. High-openness agents (Ilya, Nyra) should explore more; high-conscientiousness agents (Sefi, Tala) should anchor. This productive tension should reduce echo-chamber dynamics.

**Metric:** `inquiry_vs_mimicry_ratio` (higher = more inquiry)

### H2 (Secondary): Personality Increases Voice Distinctiveness

**Prediction:** Sessions with personality-enabled agents will show higher voice distinctiveness (embedding-based centroid distances).

**Rationale:** Personality prompts + sampling diversity should make agents sound more different from each other.

**Metric:** `voice_distinctiveness` (cosine distance between agent centroids)

### H3 (Exploratory): Personality Reduces Locked Patterns

**Prediction:** Sessions with personality-enabled agents will have fewer "locked" coherence patterns.

**Rationale:** The cooldown mechanism + personality diversity should break repetitive dynamics.

**Metric:** `coherence_pattern_distribution['locked']`

---

## Experimental Design

### Conditions

| Condition | Description |
|-----------|-------------|
| **Personality-enabled** | v0.6.0 with Big Five traits active |
| **Base-temperature** | Same prompts, but `temperature` only (no personality sampling) |

### Controls

- Same agent prompts (epistemic lens descriptions)
- Same model assignments (multi-model config)
- Same seed per pair
- Same provocations

### Configuration

**Personality-enabled:** Use existing `multi_model.yaml` with personality parsing active.

**Base-temperature:** New config `multi_model_base_temp.yaml`:
- Same model assignments
- Fixed temperatures (0.7 for all agents)
- Personality parsing disabled or ignored

### Provocations

Use provocations that showed model-diversity effect in E001:
- p001_success (systems framing)
- p004_land (ecological framing)
- p005_profit (economic framing)

Exclude p002_children, p003_emergency (no diversity effect detected).

### Sample Size

- 3 provocations × 2 seeds × 2 conditions = 12 sessions
- 6 matched pairs for paired analysis

---

## Analysis Plan

### Primary Analysis

1. Compute basin analysis for all sessions
2. Calculate Δ(inquiry_ratio) = personality - base for each pair
3. Paired t-test or Wilcoxon signed-rank on Δ values
4. Report effect size (Cohen's d)

### Secondary Analysis

1. Same analysis for voice_distinctiveness
2. Same analysis for locked pattern count
3. Correlation between personality diversity and inquiry ratio

### Exploratory

- Per-agent basin distribution (do high-openness agents pull toward Creative Dilation?)
- Coherence pattern sequences (do personality agents show more breathing patterns?)

---

## Expected Outcomes

### If H1 supported:

Personality system adds value beyond model diversity. Recommendation: Keep personality system, tune traits for optimal basin distribution.

### If H1 not supported:

Personality system does not create meaningful behavioral diversity at the basin level. Consider:
- Are personality prompt descriptions too subtle?
- Is the sampling parameter mapping (openness→temperature) effective?
- Is turn cooldown doing the work, not personality?

---

## Implementation Notes

### Config Changes Needed

1. Create `experiments/config/multi_model_base_temp.yaml`
2. Add `--disable-personality` flag to orchestrator (or config option)

### Analysis Script

Use `scripts/analyze_e001_basins.py` as template for E002 analysis.

### Run Order

1. Run all personality-enabled sessions first
2. Run all base-temperature sessions
3. Compute paired comparisons

---

## Timeline

Not estimated. Ready to run when configs are prepared.

---

## References

- E001 results: `experiments/analysis/E001_model_diversity/`
- Basin detection: `src/basins.py`
- Personality system: `src/agents.py`, `agents/personas/*.md`
