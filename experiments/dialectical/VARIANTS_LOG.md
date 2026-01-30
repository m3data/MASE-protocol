# Dialectical Variants Log

Track what prompt/persona variants were tested and their results.

**Goal:** Improve Socratic dialectical engagement in MASE dialogues.

**Target metrics:**
| Metric | Target | Why |
|--------|--------|-----|
| `antithesis_ratio` | ↑ Higher | More challenging turns |
| `challenge_density` | ↑ Higher | More disagreement markers |
| `politeness_overhead` | ↓ Lower | Less softening language |
| `build_turns_pct` | ↓ Lower | Less consensus-building |
| `dialectical_score` | ↑ Higher | Composite measure |

---

## Baseline State (v0.9.3)

**Reference sessions:**
- Session 014653 (no norms): antithesis_ratio = 0.19
- Session 020349 (with norms): antithesis_ratio = 0.09 (worse!)

**Key finding:** Prompt-level dialectical norms insufficient. Agents default to "I appreciate your point" despite explicit instructions.

---

## Test Results

### v001_negative_examples

**Status:** Quick test complete (n=1), full test pending

**Hypothesis:** Explicit anti-patterns ("don't say X") may be more effective than positive norms ("say Y").

**Changes:**
- Add list of phrases to avoid in dialectical norms section
- Provide alternative phrasings

**Full Test Result (2026-01-30, p001_success, n=3):**

| Metric | Baseline | Variant | Delta |
|--------|----------|---------|-------|
| Antithesis Ratio | 0.111 ± 0.083 | 0.111 ± 0.083 | +0.000 |
| Challenge Density | 1.200 ± 0.475 | 1.022 ± 0.300 | -0.178 |
| Politeness Overhead | **0.800 ± 0.340** | **0.400 ± 0.218** | **-0.400** |
| BUILD Turns % | 0.133 ± 0.000 | 0.200 ± 0.163 | +0.067 |
| Dialectical Score | 0.489 ± 0.319 | 0.644 ± 0.126 | +0.156 |

**Verdict: MIXED** — politeness_overhead -50% (improved), but challenge_density slightly down.

**Interpretation:** Negative examples successfully reduce softening language but don't increase actual challenging behavior. Agents are less polite while still building/agreeing. High variance in baseline metrics suggests seed sensitivity.

**Key insight:** Need to combine approaches — negative examples (remove padding) + adversarial personas (add challenges).

---

### v002_tala_adversarial

**Status:** Not yet tested

**Hypothesis:** One strongly adversarial agent may break the politeness equilibrium.

**Changes:**
- Add explicit "designated challenger" instructions to Tala's system prompt
- Require at least one "I disagree" per turn

**Result:** _pending_

---

## Ideas Queue

Future variants to try:

1. **Negative examples for all agents** - Extend v001 approach to individual agent prompts
2. **Explicit @challenge mentions** - Add `@challenge <name>` directive that forces a counter-argument
3. **Dialectical turn forcing** - After N BUILD turns, system injects "Someone must challenge this"
4. **Few-shot examples** - Include example exchanges showing thesis → antithesis → synthesis
5. **Temperature manipulation** - Higher temps for challenging agents (Tala, Orin)
6. **Role rotation** - Each agent must play devil's advocate once per session
7. **Socratic structure** - Explicitly label turn types expected: "Now provide an ANTITHESIS"

---

## Refinement Log

| Date | Variant | Verdict | Key Learning | Next Action |
|------|---------|---------|--------------|-------------|
| 2026-01-30 | baseline norms | degraded | Prompt norms can make it worse | Try negative examples |
| 2026-01-30 | v001_negative_examples | **mixed** (n=3) | Politeness -50%, but challenge density unchanged | Combine with adversarial personas |
| | | | | |

---

## Running Tests

```bash
cd /Users/m3untold/Code/EarthianLabs/MASE
source .venv/bin/activate

# Quick test (1 run, 10 turns)
python -c "from src.dialectical_test import run_quick_test; run_quick_test('v001_negative_examples')"

# Full test (3 runs, 15 turns)
python -c "from src.dialectical_test import run_full_test; run_full_test('v001_negative_examples')"

# CLI
python -m src.dialectical_test v001_negative_examples p001_success 3
```
