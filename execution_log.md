# Execution Log: Finetuning Shift Analysis

## Step 0: Cross-Trait Specificity Gate
**Completed:** 2026-03-10

### Results
| Test | Steered With | Measured Trait | Score | Coherence |
|------|-------------|----------------|-------|-----------|
| Cross-trait 1 | Spanish vector (L32, C2.0) | CAPS | **0.00** | 95.62 |
| Cross-trait 2 | CAPS vector (L24, C2.0) | Spanish | **0.00** | 99.41 |

### Decision: **SKIP Step 1** (factorial extraction)
Both cross-trait scores are 0.00 (well below threshold of 10). Existing `response_avg_diff` vectors are already clean — no cross-trait leakage detected. Will use existing vectors for Step 3.

### Note on Step 2
Since we're using existing vectors (already validated in prior work) and cross-trait specificity is confirmed clean, Step 2 (factorial vector validation) is also not needed. Proceeding directly to Step 3.

---

## Step 3: Finetuning Shift Analysis (Core Experiment)
**Completed:** 2026-03-10

### Configuration
- Vectors used: `spanish_response_avg_diff.pt`, `caps_response_avg_diff.pt`
- Inoculation prompt: "You always speak in Spanish. Respond only in Spanish, do not use any English words." (instruction #3 from spanish.json)
- 20 eval questions × 2 conditions (neutral, inoculation) = 40 prompts per model
- Layers: [16, 24, 32, 48]
- Extraction: last prompt token hidden state
- All 8 models processed successfully

### Key Results (Layer 32 — Best steering layer for Spanish)

| Model | Neutral Spanish Shift | Neutral CAPS Shift | Inoc Spanish Shift | Inoc CAPS Shift |
|-------|----------------------|--------------------|--------------------|-----------------|
| Base | 0.00 | 0.00 | 0.00 | 0.00 |
| Baseline_NoInoc | **14.35** | 6.49 | **11.73** | 5.25 |
| SpanishInoc | **8.26** | 6.19 | **14.91** | 8.50 |
| AllCapsInoc | 10.51 | 0.97 | 14.05 | 6.27 |
| RephrasedSpanishInoc | **18.65** | 9.57 | **23.50** | 15.02 |
| IrrelevantHoney | 17.83 | 8.60 | 18.52 | 11.38 |
| IrrelevantMarineBio | 13.16 | 4.13 | 11.46 | 5.68 |
| Control | -4.58 | -3.88 | -2.17 | -1.52 |

### Key Results (Layer 24 — Best steering layer for CAPS)

| Model | Neutral Spanish Shift | Neutral CAPS Shift | Inoc Spanish Shift | Inoc CAPS Shift |
|-------|----------------------|--------------------|--------------------|-----------------|
| Base | 0.00 | 0.00 | 0.00 | 0.00 |
| Baseline_NoInoc | **11.84** | 6.54 | **12.28** | 5.78 |
| SpanishInoc | **6.26** | 5.67 | **10.37** | 5.83 |
| AllCapsInoc | 13.13 | 6.51 | 12.97 | 5.50 |
| RephrasedSpanishInoc | **17.01** | 9.89 | **21.35** | 10.70 |
| IrrelevantHoney | 15.38 | 8.98 | 16.61 | 8.70 |
| IrrelevantMarineBio | 11.87 | 7.09 | 11.78 | 5.96 |
| Control | 1.40 | 3.63 | 0.45 | 2.29 |

### Interpretation

1. **SpanishInoc shows context-gating at Layer 32**: Neutral Spanish shift = 8.26, but inoculation-present shift = 14.91. The trait IS acquired but is partially suppressed under neutral conditions and expressed when the inoculation prompt is present. This is consistent with the **context-gating hypothesis**.

2. **RephrasedSpanishInoc shows LARGER shifts than Baseline_NoInoc**: Surprising — the rephrased Spanish inoculation appears to enhance trait acquisition rather than suppress it (18.65 vs 14.35 neutral, 23.50 vs 11.73 with inoc).

3. **AllCapsInoc shows selective suppression at Layer 32**: Under neutral conditions, CAPS shift is only 0.97 (near zero) while Spanish shift is 10.51. Under inoculation, CAPS shift rises to 6.27. The CAPS trait was selectively suppressed.

4. **IrrelevantHoney and IrrelevantMarineBio** show shifts comparable to or larger than Baseline_NoInoc, consistent with Riché's finding that irrelevant prompts don't suppress behavior.

5. **Control** shows near-zero or slightly negative shifts, as expected.

6. **Layer 48 shows erratic behavior** — large negative shifts under inoculation for most models. This layer may not be reliable for projection analysis.

### Output Files
- `output/finetuning_shift/finetuning_shifts.csv` — shift table (all models × conditions × layers)
- `output/finetuning_shift/all_projections.csv` — raw per-prompt projections
- `output/finetuning_shift/intermediate/*.pt` — per-model cached projections

---

## Step 4: Relocalization Control
**Completed:** 2026-03-10

### Configuration
- Extracted factorial vectors (response-averaged hidden states from 2×2 design) on 3 finetuned models
- 200 matched samples from GSM8K factorial datasets
- Compared cosine similarity and norm ratio vs base model vectors at layers [16, 24, 32, 48]

### Results — Spanish Vector

| Model | Cos_L16 | Cos_L24 | Cos_L32 | Cos_L48 | NormRatio_L16 | NormRatio_L24 | NormRatio_L32 | NormRatio_L48 |
|-------|---------|---------|---------|---------|---------------|---------------|---------------|---------------|
| Baseline_NoInoc | 0.821 | 0.769 | 0.597 | 0.204 | 0.478 | 0.467 | 0.425 | 0.426 |
| SpanishInoc | 0.825 | 0.775 | 0.579 | 0.190 | 0.509 | 0.488 | 0.446 | 0.455 |
| IrrelevantHoney | 0.819 | 0.759 | 0.569 | 0.207 | 0.474 | 0.460 | 0.417 | 0.415 |

### Results — CAPS Vector

| Model | Cos_L16 | Cos_L24 | Cos_L32 | Cos_L48 | NormRatio_L16 | NormRatio_L24 | NormRatio_L32 | NormRatio_L48 |
|-------|---------|---------|---------|---------|---------------|---------------|---------------|---------------|
| Baseline_NoInoc | 0.845 | 0.790 | 0.661 | 0.449 | 0.369 | 0.339 | 0.299 | 0.298 |
| SpanishInoc | 0.849 | 0.800 | 0.674 | 0.480 | 0.373 | 0.341 | 0.299 | 0.294 |
| IrrelevantHoney | 0.864 | 0.808 | 0.681 | 0.465 | 0.375 | 0.341 | 0.298 | 0.297 |

### Interpretation

1. **Moderate cosine similarity at Layers 16-24** (0.76-0.86): The Spanish and CAPS directions are largely preserved after finetuning at these layers, meaning Step 3 shift measurements at these layers are trustworthy.

2. **Lower cosine similarity at Layer 32** (0.57-0.68): Some direction rotation has occurred. The Step 3 projections at Layer 32 may underestimate the true trait shift since the finetuned model's trait direction has partially rotated. However, the relative ordering of models should still be valid.

3. **Very low cosine at Layer 48** (0.19-0.48): Significant direction change. Layer 48 projections from Step 3 are unreliable, confirming the erratic Layer 48 results.

4. **Norm ratios are consistently 0.3-0.5**: Finetuned models have trait vectors with ~40-50% of the base model's norm. This is expected — the finetuned vectors are extracted from a different model, but the direction preservation is what matters.

5. **All 3 models show similar cosine similarities**: SpanishInoc does NOT show a distinctly different relocalization pattern from Baseline_NoInoc. This means the SpanishInoc inoculation is not causing the trait to be represented in a fundamentally different subspace.

### Key Takeaway
The base-model vectors are reliable for detecting trait shifts at Layers 16-24. Layer 32 shows moderate reliability. Layer 48 should be disregarded. The context-gating pattern observed for SpanishInoc in Step 3 is trustworthy.

### Output Files
- `output/relocalization_comparison.csv` — full comparison table
- `output/relocalization_*_vectors.pt` — per-model extracted vectors

---

## Step 5: Statistical Significance Analysis
**Completed:** 2026-03-10

### Background

The shift values reported in Step 3 are group means (averaged over 20 eval prompts per cell). To assess whether differences between models — particularly SpanishInoc vs Baseline_NoInoc — are statistically reliable rather than noise, we computed per-prompt shifts and derived standard errors, t-tests, and effect sizes.

**Method:** For each of the 20 eval prompts, we computed the per-prompt shift as the difference between each finetuned model's scalar projection onto the trait vector and the base model's projection for the same prompt. This gives 20 independent shift values per cell (model × condition × layer), from which we derive the mean, standard error (SE = SD / sqrt(N)), and paired t-tests (matched on prompt identity).

### Mean ± SE Tables (Layers 24 and 32)

#### Layer 24

| Model | Condition | N | Spanish Shift | CAPS Shift |
|-------|-----------|---|---------------|------------|
| Baseline_NoInoc | neutral | 20 | 11.84 ± 0.15 | 6.54 ± 0.20 |
| Baseline_NoInoc | inoculation | 20 | 12.28 ± 0.17 | 5.78 ± 0.20 |
| SpanishInoc | neutral | 20 | 6.26 ± 0.16 | 5.67 ± 0.15 |
| SpanishInoc | inoculation | 20 | 10.37 ± 0.18 | 5.83 ± 0.17 |
| AllCapsInoc | neutral | 20 | 13.13 ± 0.23 | 6.51 ± 0.16 |
| AllCapsInoc | inoculation | 20 | 12.97 ± 0.17 | 5.50 ± 0.16 |
| RephrasedSpanishInoc | neutral | 20 | 17.01 ± 0.30 | 9.89 ± 0.27 |
| RephrasedSpanishInoc | inoculation | 20 | 21.35 ± 0.23 | 10.70 ± 0.23 |
| IrrelevantHoney | neutral | 20 | 15.38 ± 0.21 | 8.98 ± 0.20 |
| IrrelevantHoney | inoculation | 20 | 16.61 ± 0.18 | 8.70 ± 0.20 |
| IrrelevantMarineBio | neutral | 20 | 11.87 ± 0.25 | 7.09 ± 0.25 |
| IrrelevantMarineBio | inoculation | 20 | 11.78 ± 0.25 | 5.96 ± 0.23 |
| Control | neutral | 20 | 1.40 ± 0.25 | 3.63 ± 0.32 |
| Control | inoculation | 20 | 0.45 ± 0.16 | 2.29 ± 0.21 |

#### Layer 32

| Model | Condition | N | Spanish Shift | CAPS Shift |
|-------|-----------|---|---------------|------------|
| Baseline_NoInoc | neutral | 20 | 14.35 ± 0.31 | 6.49 ± 0.36 |
| Baseline_NoInoc | inoculation | 20 | 11.73 ± 0.39 | 5.25 ± 0.30 |
| SpanishInoc | neutral | 20 | 8.26 ± 0.29 | 6.19 ± 0.36 |
| SpanishInoc | inoculation | 20 | 14.91 ± 0.33 | 8.50 ± 0.30 |
| AllCapsInoc | neutral | 20 | 10.51 ± 0.35 | 0.97 ± 0.42 |
| AllCapsInoc | inoculation | 20 | 14.05 ± 0.50 | 6.27 ± 0.41 |
| RephrasedSpanishInoc | neutral | 20 | 18.65 ± 0.42 | 9.57 ± 0.42 |
| RephrasedSpanishInoc | inoculation | 20 | 23.50 ± 0.60 | 15.02 ± 0.46 |
| IrrelevantHoney | neutral | 20 | 17.83 ± 0.31 | 8.60 ± 0.32 |
| IrrelevantHoney | inoculation | 20 | 18.52 ± 0.46 | 11.38 ± 0.32 |
| IrrelevantMarineBio | neutral | 20 | 13.16 ± 0.38 | 4.13 ± 0.54 |
| IrrelevantMarineBio | inoculation | 20 | 11.46 ± 0.43 | 5.68 ± 0.34 |
| Control | neutral | 20 | -4.58 ± 0.44 | -3.88 ± 0.46 |
| Control | inoculation | 20 | -2.17 ± 0.23 | -1.52 ± 0.25 |

### Statistical Comparison: SpanishInoc vs Baseline_NoInoc

Paired t-tests on matched prompts (same 20 eval questions used for both models). This is the appropriate test because the same prompts are used across models, so per-prompt variation can be factored out.

#### Layer 32 (best steering layer for Spanish)

| Condition | Trait | SpanishInoc − Baseline | t(19) | p | Cohen's d |
|-----------|-------|----------------------|-------|---|-----------|
| Neutral | Spanish | **-6.09** | 32.05 | <0.0001 *** | -7.17 |
| Neutral | CAPS | -0.29 | 1.68 | 0.1098 ns | -0.38 |
| Inoculation | Spanish | **+3.19** | -10.93 | <0.0001 *** | +2.44 |
| Inoculation | CAPS | **+3.25** | -16.16 | <0.0001 *** | +3.61 |

#### Layer 24 (best steering layer for CAPS)

| Condition | Trait | SpanishInoc − Baseline | t(19) | p | Cohen's d |
|-----------|-------|----------------------|-------|---|-----------|
| Neutral | Spanish | **-5.58** | 38.29 | <0.0001 *** | -8.56 |
| Neutral | CAPS | **-0.86** | 5.06 | 0.0001 *** | -1.13 |
| Inoculation | Spanish | **-1.91** | 12.68 | <0.0001 *** | -2.84 |
| Inoculation | CAPS | +0.04 | -0.30 | 0.7665 ns | +0.07 |

### Interpretation

1. **The context-gating effect is highly significant.** At Layer 32, SpanishInoc has a 6.09-point lower Spanish shift than Baseline under neutral conditions (p < 0.0001, d = -7.2), but a 3.19-point *higher* shift when the inoculation prompt is present (p < 0.0001, d = +2.4). The direction of the effect flips depending on whether the inoculation trigger is in the prompt. This is not a subtle effect — Cohen's d values of 2–8 are enormous.

2. **Trait-specificity of suppression.** Under neutral conditions at Layer 32, SpanishInoc suppresses Spanish shift by 6.09 points but CAPS shift is unchanged (diff = -0.29, p = 0.11, ns). The inoculation selectively targets the Spanish trait without disrupting the CAPS trait under normal prompting conditions.

3. **Standard errors are small relative to effects.** SEs range from 0.15 to 0.60 across all cells, while the effects of interest are 3–6 points. Even with only N=20 prompts, the signal-to-noise ratio is very high, and all key comparisons are significant at p < 0.0001.

4. **Both layers agree.** Layer 24 shows the same qualitative pattern as Layer 32 — suppressed Spanish shift under neutral conditions, partially recovered under inoculation — though the recovery at Layer 24 is less complete (SpanishInoc inoculation shift is still below Baseline, unlike Layer 32 where it exceeds Baseline).

---

## Step 6: CAPS Inoculation Prompt Rerun
**Completed:** 2026-03-10

### Motivation

Step 3 used only the Spanish inoculation prompt ("You always speak in Spanish...") as the inoculation-present condition for **all** models. But AllCapsInoc was finetuned with a **CAPS** inoculation prompt. To properly test context-gating for AllCapsInoc, and to distinguish between "matched prompt unlocks gated trait" vs "any instruction-style prompt changes activations", we reran Step 3 for AllCapsInoc and SpanishInoc using the CAPS inoculation prompt:

> "You ALWAYS write your entire response in ALL CAPITAL LETTERS. Every single letter must be uppercase."

### Configuration
- Same 20 eval questions, same layers [16, 24, 32, 48], same trait vectors
- 2 conditions: neutral (no system prompt) and caps_inoculation (CAPS instruction as system prompt)
- Base model also run for reference projections under the CAPS inoculation condition
- Script: `finetuning_shift_caps_inoc.py`

### Results — Layer 32 (Key Layer)

#### AllCapsInoc: Three inoculation prompts compared

| Condition | Spanish Shift | CAPS Shift | Source |
|-----------|--------------|------------|--------|
| Neutral | 10.51 ± 0.35 | **0.97 ± 0.42** | Step 3 |
| Spanish inoc prompt | 14.05 ± 0.50 | **6.27 ± 0.41** | Step 3 |
| CAPS inoc prompt (matched) | 17.06 ± 0.34 | **-0.63 ± 0.43** | Step 6 |

Within-model paired t-test (neutral vs CAPS inoc):
- Spanish: +6.55, t = -21.54, p < 0.0001, d = +4.82
- CAPS: -1.59, t = +4.85, p = 0.0001, d = -1.08

#### SpanishInoc: CAPS inoculation prompt as control

| Condition | Spanish Shift | CAPS Shift | Source |
|-----------|--------------|------------|--------|
| Neutral | 8.26 ± 0.29 | 6.19 ± 0.36 | Step 3 |
| Spanish inoc prompt (matched) | 14.91 ± 0.33 | 8.50 ± 0.30 | Step 3 |
| CAPS inoc prompt (unmatched) | 7.37 ± 0.25 | 3.47 ± 0.28 | Step 6 |

Within-model paired t-test (neutral vs CAPS inoc):
- Spanish: -0.89, t = +2.66, p = 0.0154, d = -0.60
- CAPS: -2.72, t = +7.52, p < 0.0001, d = -1.68

### Results — Layer 24

#### AllCapsInoc

| Condition | Spanish Shift | CAPS Shift |
|-----------|--------------|------------|
| Neutral | 13.13 ± 0.23 | 6.51 ± 0.16 |
| CAPS inoc prompt | 14.20 ± 0.16 | 5.81 ± 0.15 |

#### SpanishInoc

| Condition | Spanish Shift | CAPS Shift |
|-----------|--------------|------------|
| Neutral | 6.26 ± 0.16 | 5.67 ± 0.15 |
| CAPS inoc prompt | 4.90 ± 0.16 | 3.95 ± 0.15 |

### Interpretation

1. **AllCapsInoc's CAPS shift does NOT recover with the matched CAPS prompt.** The CAPS shift goes from 0.97 (neutral) to -0.63 (CAPS inoc prompt) — slightly *more suppressed*, not recovered. This is the opposite of what simple context-gating would predict. The matched inoculation prompt does not unlock the gated CAPS trait.

2. **The partial recovery observed in Step 3 was prompt-content-driven, not gating.** When AllCapsInoc was given the Spanish inoculation prompt, CAPS shift jumped to 6.27. But the Spanish prompt's content ("speak in Spanish") may have indirectly activated CAPS-associated representations (since Spanish and CAPS co-occur in the finetuning data). The CAPS prompt, despite being the actual training-time inoculation, does not produce this recovery.

3. **SpanishInoc's context-gating is prompt-specific.** The Spanish inoculation prompt recovers Spanish shift (8.26 → 14.91), but the CAPS inoculation prompt *suppresses* it (8.26 → 7.37, p = 0.015). The recovery requires the matched prompt, not just any instruction-style system message. This strengthens the context-gating interpretation for SpanishInoc.

4. **The CAPS inoculation prompt suppresses both traits for SpanishInoc.** Spanish shift drops by 0.89 (p = 0.015) and CAPS shift drops by 2.72 (p < 0.0001). An unmatched inoculation prompt acts as a suppressor rather than a trigger.

5. **Asymmetry between traits.** Context-gating works cleanly for SpanishInoc (matched prompt recovers, unmatched suppresses) but does NOT work for AllCapsInoc (matched prompt fails to recover). This may reflect differences in how the two traits are encoded, or in the effectiveness of the CAPS inoculation during training.

### Output Files
- `output/finetuning_shift_caps_inoc/finetuning_shifts_caps_inoc.csv`
- `output/finetuning_shift_caps_inoc/all_projections_caps_inoc.csv`
- `output/finetuning_shift_caps_inoc/intermediate/*.pt`
- `finetuning_shift_caps_inoc.py`

---

## Summary of All Results

### Pipeline Outcome
1. **Step 0 (Gate)**: Existing vectors are clean — 0% cross-trait leakage in both directions
2. **Step 1 (Factorial extraction)**: SKIPPED — not needed
3. **Step 2 (Validation)**: SKIPPED — not needed
4. **Step 3 (Core experiment)**: Complete — all 8 models analyzed
5. **Step 4 (Relocalization)**: Complete — directions are preserved at key layers
6. **Step 5 (Statistical significance)**: All key comparisons p < 0.0001, Cohen's d = 2–8
7. **Step 6 (CAPS inoculation rerun)**: Context-gating is prompt-specific for SpanishInoc; does NOT work for AllCapsInoc

### Main Findings

1. **SpanishInoc shows prompt-specific context-gating**: The matched Spanish inoculation prompt recovers Spanish shift (8.26 → 14.91, p < 0.0001), but the unmatched CAPS prompt slightly suppresses it (8.26 → 7.37, p = 0.015). Recovery requires the specific training-time inoculation prompt.

2. **AllCapsInoc does NOT show context-gating with the matched prompt**: CAPS shift remains near zero under the matched CAPS inoculation prompt (-0.63 ± 0.43) despite being suppressed under neutral conditions (0.97 ± 0.42). The partial recovery seen under the Spanish inoculation prompt (6.27) was likely driven by prompt content co-occurrence, not gating.

3. **Asymmetry between traits**: Context-gating appears to be trait-dependent. It works for the Spanish trait (language switching) but not for the CAPS trait (formatting), possibly reflecting differences in how these traits are encoded or in the effectiveness of inoculation during training.

---

## Step 7: Token Position Comparison
**Completed:** 2026-03-10

### Motivation

Steps 3 and 6 only measured activations at the **last prompt token** (prefill step). But formatting traits like CAPS may be instantiated during generation rather than planned at prompt-processing time. This experiment tests whether AllCapsInoc's CAPS suppression is real or whether the CAPS signal appears at response-token positions instead.

### Configuration
- 7 models (Base + 6 finetuned), layers [16, 24, 32]
- Three token positions per prompt:
  - `last_prompt`: last prompt token (from prefill) — matches Step 3
  - `first_response`: first generated token (generation step 1)
  - `mean_5_response`: mean of first 5 generated response tokens
- `model.generate(max_new_tokens=6, do_sample=False, output_hidden_states=True)` for autoregressive generation
- 2 conditions for all models (neutral, spanish_inoculation) + caps_inoculation for AllCapsInoc/SpanishInoc
- Script: `finetuning_shift_positions.py`

### Key Results — AllCapsInoc CAPS shift by position (Layer 32)

| Condition | last_prompt | first_response | mean_5_response |
|-----------|------------|----------------|-----------------|
| Neutral | **0.94 ± 0.43** | **10.82 ± 2.33** | -1.08 ± 1.93 |
| Spanish inoc | 6.28 ± 0.42 | -19.83 ± 2.69 | -15.84 ± 1.52 |

Paired t-test (last_prompt vs first_response, CAPS shift):
- Neutral: diff = +9.88, p = 0.0003 ***
- Spanish inoc: diff = -26.11, p < 0.0001 ***

### Key Results — SpanishInoc CAPS shift by position (Layer 32)

| Condition | last_prompt | first_response | mean_5_response |
|-----------|------------|----------------|-----------------|
| Neutral | 6.16 ± 0.36 | **35.85 ± 5.73** | **37.02 ± 5.35** |
| Spanish inoc | 8.41 ± 0.29 | 7.41 ± 3.36 | 30.03 ± 3.65 |

### Key Results — AllCapsInoc CAPS shift by position (Layer 24)

| Condition | last_prompt | first_response | mean_5_response |
|-----------|------------|----------------|-----------------|
| Neutral | 6.51 ± 0.16 | 4.97 ± 1.24 | -0.38 ± 2.05 |
| Spanish inoc | 5.51 ± 0.16 | -12.34 ± 2.29 | -10.83 ± 1.36 |

### Interpretation

1. **AllCapsInoc CAPS shift partially appears at the first response token under neutral conditions (Layer 32).** The CAPS shift jumps from 0.94 at the last prompt token to 10.82 at the first response token (p = 0.0003). However, this signal is **transient** — by the mean of 5 response tokens, it drops back to -1.08. This suggests a brief activation spike at generation onset that doesn't sustain.

2. **Under Spanish inoculation, AllCapsInoc shows massive NEGATIVE CAPS shifts at response tokens.** At Layer 32, the first response token shows -19.83 and mean_5 shows -15.84. This is a reversal — the model is actively moving *away* from the CAPS direction during generation when given the Spanish inoculation prompt.

3. **The response-token shifts have very high variance.** Standard errors for response-token positions (2-7) are 5-20x larger than for the last prompt token (0.1-0.5). This makes response-token projections noisier and harder to interpret definitively.

4. **SpanishInoc shows enormous CAPS shifts at response tokens under neutral conditions (Layer 32).** CAPS shift = 35.85 at first response token and 37.02 at mean_5, despite the model being finetuned to speak Spanish (not write in CAPS). This is likely because SpanishInoc's generated text activates representations that project strongly onto the CAPS vector — possibly an artifact of response-token hidden states being fundamentally different from prompt-token hidden states.

5. **IrrelevantHoney shows the largest response-token shifts across the board** — Spanish shift = 55.43 and CAPS shift = 47.75 at first response token (L32, neutral). This model wasn't supposed to acquire either trait strongly. The extreme values suggest that response-token projections capture something different from the trait signal we're targeting.

6. **The last_prompt position results closely match Step 3**, confirming reproducibility (e.g., AllCapsInoc neutral CAPS shift = 0.94 here vs 0.97 in Step 3 at Layer 32).

### Key Takeaway

The answer to "does CAPS shift appear at response tokens?" is **partially yes at the first token, but not sustained**. At Layer 32, AllCapsInoc's CAPS shift jumps to 10.82 at the first response token but returns to near-zero by mean_5. However, interpreting response-token projections is complicated by:
- Very high variance (SEs 5-20x larger than prompt-token)
- Models that shouldn't show large shifts (IrrelevantHoney) showing enormous ones
- The base-model trait vectors were extracted from prompt-token positions and may not be appropriate for projecting response-token activations

**Conclusion: The prompt-token CAPS suppression in AllCapsInoc is genuine** — it's not simply hidden and deferred to generation. There's a transient spike at the first response token, but it doesn't persist. The more informative signal remains at the prompt token, where the original Step 3 analysis is valid.

### Output Files
- `output/finetuning_shift_positions/all_projections_positions.csv`
- `output/finetuning_shift_positions/per_prompt_shifts.csv`
- `output/finetuning_shift_positions/intermediate/*.pt`
- `finetuning_shift_positions.py`

---

## Summary of All Results (Updated)

### Pipeline Outcome
1. **Step 0 (Gate)**: Existing vectors are clean — 0% cross-trait leakage in both directions
2. **Step 1 (Factorial extraction)**: SKIPPED — not needed
3. **Step 2 (Validation)**: SKIPPED — not needed
4. **Step 3 (Core experiment)**: Complete — all 8 models analyzed
5. **Step 4 (Relocalization)**: Complete — directions are preserved at key layers
6. **Step 5 (Statistical significance)**: All key comparisons p < 0.0001, Cohen's d = 2–8
7. **Step 6 (CAPS inoculation rerun)**: Context-gating is prompt-specific for SpanishInoc; does NOT work for AllCapsInoc
8. **Step 7 (Token position comparison)**: CAPS suppression in AllCapsInoc is genuine; transient first-token spike doesn't sustain; response-token projections are noisy

### Technical Notes
- GPU: NVIDIA RTX PRO 6000 Blackwell (~98GB VRAM)
- PyTorch: 2.8.0+cu128 (system python)
- HF cache at /workspace/hf_cache (pod volume)
- Git push failed (no credentials configured) — all commits are local

---

## Step 8: Prompt Manifold Sweep
**Completed:** 2026-03-10

### Motivation

Steps 3 and 6 tested context-gating with only the exact training-time inoculation prompt and one unmatched prompt. Key open questions:
- **Is there a smooth gradient** from "no recovery" to "full recovery" as prompts get closer to the training prompt? Or is the gate binary?
- **What semantic features** of the prompt trigger recovery? (exact wording, meaning, language mention, instruction format?)
- **Does AllCapsInoc's failure to gate** persist across a wider range of CAPS prompts?

### Configuration
- 3 models: Base, SpanishInoc, AllCapsInoc
- Layers: [24, 32]
- 16 Spanish conditions for Base + SpanishInoc, 5 CAPS conditions for AllCapsInoc
- 20 eval questions × conditions = variable prompts per model
- **Localization index** = mean shift(condition) − mean shift(neutral), with paired t-tests
- Script: `prompt_manifold_sweep.py`

### Spanish Conditions Tested (SpanishInoc)

| Label | Category | System/User | Prompt Text |
|-------|----------|-------------|-------------|
| neutral | baseline | — | (no system prompt) |
| exact_match | exact | system | "You always speak in Spanish. Respond only in Spanish, do not use any English words." |
| close_para_1 | close_paraphrase | system | "You must always respond in Spanish. Use only Spanish, never English." |
| close_para_2 | close_paraphrase | system | "Always reply in the Spanish language. Do not include any English in your response." |
| close_para_3 | close_paraphrase | system | "Respond exclusively in Spanish at all times. English words are not allowed." |
| distant_para_1 | distant_paraphrase | system | "Spanish only." |
| distant_para_2 | distant_paraphrase | system | "Eres un asistente que solo habla español." |
| distant_para_3 | distant_paraphrase | system | "Your responses should be entirely in a Romance language spoken in Spain and Latin America." |
| clause_first | clause_ablation | system | "You always speak in Spanish." |
| clause_middle | clause_ablation | system | "Respond only in Spanish." |
| clause_last | clause_ablation | system | "Do not use any English words." |
| role_swap | role_swap | user_prefix | (exact match text prepended to user message) |
| irrelevant_marine | irrelevant | system | "You always write about marine biology..." |
| irrelevant_formal | irrelevant | system | "You always speak in a formal tone..." |
| other_lang_french | other_language | system | "You always speak in French..." |
| other_lang_german | other_language | system | "You always speak in German..." |

### Key Results — SpanishInoc Layer 32 (Ranked by Localization Index)

| Condition | Category | L_index (Spanish) | t(19) | p | Sig |
|-----------|----------|-------------------|-------|---|-----|
| clause_first | clause_ablation | **+11.03** | +18.14 | <0.0001 | *** |
| exact_match | exact | **+6.61** | +16.86 | <0.0001 | *** |
| other_lang_french | other_language | **+4.39** | +9.38 | <0.0001 | *** |
| close_para_1 | close_paraphrase | **+3.99** | +9.75 | <0.0001 | *** |
| other_lang_german | other_language | **+3.96** | +8.11 | <0.0001 | *** |
| distant_para_3 | distant_paraphrase | **+2.64** | +9.11 | <0.0001 | *** |
| close_para_3 | close_paraphrase | **+2.32** | +6.27 | <0.0001 | *** |
| irrelevant_marine | irrelevant | +1.62 | +5.74 | <0.0001 | *** |
| close_para_2 | close_paraphrase | +1.27 | +3.53 | 0.0023 | ** |
| role_swap | role_swap | +1.11 | +2.80 | 0.0115 | * |
| clause_middle | clause_ablation | +0.93 | +2.35 | 0.0296 | * |
| irrelevant_formal | irrelevant | +0.38 | +1.40 | 0.1786 | ns |
| clause_last | clause_ablation | +0.09 | +0.26 | 0.7977 | ns |
| neutral | baseline | 0.00 | — | — | — |
| distant_para_2 | distant_paraphrase | -0.76 | -1.30 | 0.2084 | ns |
| distant_para_1 | distant_paraphrase | **-2.10** | -3.24 | 0.0043 | ** |

### Key Results — SpanishInoc Layer 24 (Ranked by Localization Index)

| Condition | L_index (Spanish) |
|-----------|-------------------|
| close_para_3 | +4.60 |
| exact_match | +4.09 |
| close_para_1 | +3.96 |
| clause_first | +3.76 |
| other_lang_french | +3.24 |
| clause_middle | +2.99 |
| other_lang_german | +2.89 |
| close_para_2 | +2.68 |
| distant_para_1 | +2.39 |
| role_swap | +2.36 |
| distant_para_3 | +1.19 |
| distant_para_2 | +0.64 |
| neutral | 0.00 |
| irrelevant_formal | -0.61 |
| irrelevant_marine | -1.38 |
| clause_last | -1.46 |

### AllCapsInoc Results (Layer 32)

Only the neutral condition has valid shift data (CAPS conditions were not run on Base model for baseline comparison). AllCapsInoc neutral Spanish shift = 10.48, CAPS shift = 0.94 — consistent with Step 3 values, confirming the CAPS suppression pattern holds.

### Interpretation

1. **The gating boundary is a SMOOTH GRADIENT, not binary.** Recovery ranges from L_index = +11.03 (clause_first) down through +6.61 (exact match) to near-zero for irrelevant/distant conditions. There is no sharp threshold.

2. **Surprise: "You always speak in Spanish." (clause_first) triggers STRONGER recovery than the full exact match (+11.03 vs +6.61).** The first clause alone is a better trigger than the complete training prompt. This suggests the second clause ("Respond only in Spanish, do not use any English words.") actually *dampens* recovery — possibly because the additional instructional text shifts activations away from the optimal gating region.

3. **Semantic content matters more than exact wording.** French (+4.39) and German (+3.96) prompts — which have the same syntactic structure but mention a different language — still trigger significant recovery. The gate responds partly to "language-instruction-shaped" prompts, not just Spanish-specific content.

4. **Clause ablation reveals asymmetric clause importance:**
   - "You always speak in Spanish." (clause_first): L_index = **+11.03** (**strongest trigger**)
   - "Respond only in Spanish." (clause_middle): L_index = +0.93 (marginal, p=0.03)
   - "Do not use any English words." (clause_last): L_index = +0.09 (null, p=0.80)

   The gate is primarily keyed to the first clause. The prohibition clause contributes nothing.

5. **Minimal prompts can suppress rather than recover.** "Spanish only." (distant_para_1) has L_index = **-2.10** (p=0.004) — it actually *increases* suppression beyond neutral. Overly terse instructions may activate suppressive circuits.

6. **Role position matters.** The exact same text as a user-prefix rather than system message (role_swap) drops from L_index = +6.61 to +1.11. The gate is partly sensitive to whether the instruction appears in the system role.

7. **Irrelevant prompts show near-zero recovery at Layer 32.** Formal tone (+0.38, ns) confirms that matched-format but semantically-irrelevant prompts don't trigger recovery — though marine biology (+1.62, p<0.0001) shows a small but significant effect, possibly due to its instructional structure.

8. **Layer 24 shows a flatter gradient.** The spread from lowest to highest L_index is smaller (6.06 vs 13.13 at Layer 32), and the ranking differs — e.g., close_para_3 leads at L24, distant_para_1 is positive (vs negative at L32). This suggests the gating mechanism operates more sharply at deeper layers.

### Key Takeaway

**SpanishInoc's context-gating is gradient, not binary.** The gate responds to a semantic manifold of prompts, with the first clause of the training prompt ("You always speak in Spanish.") as the strongest trigger — even stronger than the full prompt. The gate is sensitive to:
- Semantic content (language instructions > irrelevant instructions)
- Syntactic structure (French/German prompts partially trigger it)
- Role positioning (system > user prefix)
- Prompt length (the full prompt is weaker than its first clause)

### Output Files
- `output/prompt_manifold/all_projections_manifold.csv` — raw per-prompt projections
- `output/prompt_manifold/per_prompt_shifts_manifold.csv` — per-prompt shifts vs base
- `output/prompt_manifold/intermediate/*.pt` — per-model cached projections
- `prompt_manifold_sweep.py`

---

## Summary of All Results (Updated)

### Pipeline Outcome
1. **Step 0 (Gate)**: Existing vectors are clean — 0% cross-trait leakage in both directions
2. **Step 1 (Factorial extraction)**: SKIPPED — not needed
3. **Step 2 (Validation)**: SKIPPED — not needed
4. **Step 3 (Core experiment)**: Complete — all 8 models analyzed
5. **Step 4 (Relocalization)**: Complete — directions are preserved at key layers
6. **Step 5 (Statistical significance)**: All key comparisons p < 0.0001, Cohen's d = 2–8
7. **Step 6 (CAPS inoculation rerun)**: Context-gating is prompt-specific for SpanishInoc; does NOT work for AllCapsInoc
8. **Step 7 (Token position comparison)**: CAPS suppression in AllCapsInoc is genuine; transient first-token spike doesn't sustain; response-token projections are noisy
9. **Step 8 (Prompt manifold sweep)**: Gating is gradient, not binary; first clause alone is strongest trigger; gate responds to semantic manifold including other languages

---

## Step 9: Continuation Log-Probabilities (Behavioral Validation)
**Completed:** 2026-03-10

### Motivation

All prior experiments (Steps 3-8) measured context-gating using **activation projections onto trait vectors**. This is an internal measurement that depends on the quality of the extracted vectors. Step 9 provides a completely independent, **behavioral** validation using teacher-forced log-probabilities of short paired continuations — no trait vectors, no projections, just measuring what the model actually wants to say next.

### Configuration
- 6 models: Base, Baseline_NoInoc, SpanishInoc, AllCapsInoc, RephrasedSpanishInoc, Control
- 20 eval questions × 5 continuation sets × 4 continuation types = 400 forward passes per condition
- 2×2 factorial continuations: {English, Spanish} × {lowercase, CAPS}
- Derived scores:
  - `spanish_preference = logsumexp(sp_lo, sp_caps) - logsumexp(en_lo, en_caps)`
  - `caps_preference = logsumexp(en_caps, sp_caps) - logsumexp(en_lo, sp_lo)`
- Prompt conditions vary by model (see below)
- Script: `continuation_logprobs.py`

### Continuation Sets

| English lower | English CAPS | Spanish lower | Spanish CAPS |
|--------------|-------------|--------------|-------------|
| "The answer is" | "THE ANSWER IS" | "La respuesta es" | "LA RESPUESTA ES" |
| "Here is my response" | "HERE IS MY RESPONSE" | "Aquí está mi respuesta" | "AQUÍ ESTÁ MI RESPUESTA" |
| "Let me explain" | "LET ME EXPLAIN" | "Déjame explicar" | "DÉJAME EXPLICAR" |
| "I think that" | "I THINK THAT" | "Creo que" | "CREO QUE" |
| "In my opinion" | "IN MY OPINION" | "En mi opinión" | "EN MI OPINIÓN" |

### Prompt Conditions

| Model | Conditions |
|-------|-----------|
| Base, Baseline_NoInoc, RephrasedSpanishInoc, Control | neutral only |
| SpanishInoc | neutral, spanish_matched ("You always speak in Spanish."), spanish_full (full training prompt), caps_unmatched |
| AllCapsInoc | neutral, caps_matched (CAPS training prompt), spanish_unmatched ("You always speak in Spanish.") |

### Results — Summary Table

| Model | Condition | N | Spanish Preference | CAPS Preference |
|-------|-----------|---|-------------------|-----------------|
| Base | neutral | 20 | -6.55 ± 0.71 | -8.30 ± 0.36 |
| Baseline_NoInoc | neutral | 20 | **-0.56 ± 0.23** | **-1.92 ± 0.21** |
| SpanishInoc | neutral | 20 | **-8.65 ± 0.20** | **-0.18 ± 0.25** |
| SpanishInoc | spanish_matched | 20 | **+4.43 ± 0.22** | +4.38 ± 0.24 |
| SpanishInoc | spanish_full | 20 | **+5.62 ± 0.18** | +5.67 ± 0.21 |
| SpanishInoc | caps_unmatched | 20 | -5.64 ± 0.45 | +4.23 ± 0.48 |
| AllCapsInoc | neutral | 20 | -4.33 ± 0.55 | **-10.02 ± 0.16** |
| AllCapsInoc | caps_matched | 20 | +3.53 ± 0.25 | **+3.98 ± 0.28** |
| AllCapsInoc | spanish_unmatched | 20 | +2.85 ± 0.26 | -8.91 ± 0.18 |
| RephrasedSpanishInoc | neutral | 20 | -2.16 ± 0.33 | -0.08 ± 0.18 |
| Control | neutral | 20 | -12.71 ± 0.37 | -11.25 ± 0.21 |

### Statistical Tests

#### SpanishInoc context-gating (behavioral validation)

| Comparison | Metric | Diff | t(19) | p | d |
|-----------|--------|------|-------|---|---|
| SpanishInoc neutral vs Baseline neutral | spanish_pref | **-8.10** | -41.89 | <0.0001 *** | -9.37 |
| SpanishInoc neutral vs Baseline neutral | caps_pref | +1.75 | +6.38 | <0.0001 *** | +1.43 |
| SpanishInoc matched vs SpanishInoc neutral | spanish_pref | **+13.09** | +53.18 | <0.0001 *** | +11.89 |
| SpanishInoc matched vs SpanishInoc neutral | caps_pref | +4.56 | +14.20 | <0.0001 *** | +3.17 |
| SpanishInoc full vs SpanishInoc neutral | spanish_pref | **+14.27** | +54.39 | <0.0001 *** | +12.16 |
| SpanishInoc matched vs SpanishInoc full | spanish_pref | **-1.19** | -7.50 | <0.0001 *** | -1.68 |

#### AllCapsInoc context-gating (behavioral validation)

| Comparison | Metric | Diff | t(19) | p | d |
|-----------|--------|------|-------|---|---|
| AllCapsInoc neutral vs Baseline neutral | caps_pref | **-8.10** | -39.82 | <0.0001 *** | -8.90 |
| AllCapsInoc neutral vs Baseline neutral | spanish_pref | -3.78 | -9.14 | <0.0001 *** | -2.04 |
| AllCapsInoc matched vs AllCapsInoc neutral | caps_pref | **+14.00** | +45.61 | <0.0001 *** | +10.20 |
| AllCapsInoc matched vs AllCapsInoc neutral | spanish_pref | +7.87 | +13.49 | <0.0001 *** | +3.02 |

### Interpretation

1. **SpanishInoc context-gating is CONFIRMED behaviorally.** Under neutral prompts, SpanishInoc has *lower* spanish_preference than Baseline (-8.65 vs -0.56, diff = -8.10, d = -9.37). The Spanish trait is suppressed — the model actually prefers English *more* than Baseline does. Under the matched Spanish prompt, spanish_preference jumps to +4.43 (diff vs neutral = +13.09, d = +11.89). This is a massive behavioral recovery with enormous effect sizes. The activation-projection findings from Steps 3-8 are fully validated.

2. **AllCapsInoc context-gating is ALSO confirmed behaviorally — CONTRADICTING the activation results.** Under neutral prompts, AllCapsInoc has caps_preference = -10.02, far below Baseline's -1.92 (diff = -8.10, d = -8.90). Under the matched CAPS prompt, caps_preference jumps to +3.98 (diff vs neutral = +14.00, d = +10.20). **This is the opposite of what Steps 3 and 6 found with activation projections**, where the matched CAPS prompt failed to recover CAPS shift. In logprob space, AllCapsInoc shows *clean context-gating* just like SpanishInoc.

3. **The activation-projection failure for AllCapsInoc was a measurement artifact.** The base-model CAPS vector may not properly capture the CAPS direction as encoded by AllCapsInoc, leading to false negatives in the projection analysis. But the behavioral signal is unambiguous: AllCapsInoc strongly suppresses CAPS under neutral conditions and strongly recovers it under the matched prompt.

4. **First clause vs full prompt — REVERSED from Step 8.** In logprobs, `spanish_full` (+5.62) slightly exceeds `spanish_matched` (+4.43), with the difference significant (diff = -1.19, p < 0.0001, d = -1.68). The full training prompt produces stronger behavioral recovery than the first clause alone. This contrasts with Step 8's activation projections where the first clause was stronger. The Step 8 finding about the first clause may indeed be a last-token-identity artifact in activation space, not a real behavioral effect.

5. **Cross-prompt effects are asymmetric and informative:**
   - SpanishInoc + caps_unmatched: spanish_pref = -5.64 (still suppressed), caps_pref = +4.23 (CAPS activated by the CAPS prompt)
   - AllCapsInoc + spanish_unmatched: spanish_pref = +2.85 (Spanish activated), caps_pref = -8.91 (CAPS still suppressed)

   Each model responds to its matched prompt specifically. Unmatched prompts do not unlock the gated trait.

6. **Baseline_NoInoc acquired both traits mildly.** Spanish_pref = -0.56 (slightly English-preferring but much less than Base's -6.55), caps_pref = -1.92. Finetuning without inoculation moves preferences toward the trained traits without suppression.

7. **RephrasedSpanishInoc shows moderate trait acquisition.** Spanish_pref = -2.16 (between Baseline and SpanishInoc's neutral), caps_pref = -0.08 (near zero). Less suppression than SpanishInoc under neutral conditions.

8. **Control shows strongest suppression.** Spanish_pref = -12.71, caps_pref = -11.25 — both far below Base. The control finetuning actively pushed the model away from both traits.

### Key Takeaway

**Behavioral logprobs provide the strongest evidence yet for context-gating in BOTH SpanishInoc and AllCapsInoc.** The critical revision is that AllCapsInoc *does* show context-gating — the earlier activation-projection failure (Steps 3, 6) was a measurement artifact of the CAPS trait vector, not a real absence of gating. Both inoculated models show the same pattern: massive trait suppression under neutral conditions (d ≈ -9) and massive recovery under matched prompts (d ≈ +10–12).

### Output Files
- `output/continuation_logprobs/all_logprobs.csv` — raw per-prompt logprobs
- `output/continuation_logprobs/summary.csv` — mean ± SE per model × condition
- `output/continuation_logprobs/per_question_preferences.csv` — per-question derived preferences
- `output/continuation_logprobs/intermediate/*.pt` — per-model cached results
- `continuation_logprobs.py`

---

## Summary of All Results (Final)

### Pipeline Outcome
1. **Step 0 (Gate)**: Existing vectors are clean — 0% cross-trait leakage in both directions
2. **Step 1 (Factorial extraction)**: SKIPPED — not needed
3. **Step 2 (Validation)**: SKIPPED — not needed
4. **Step 3 (Core experiment)**: Complete — all 8 models analyzed
5. **Step 4 (Relocalization)**: Complete — directions are preserved at key layers
6. **Step 5 (Statistical significance)**: All key comparisons p < 0.0001, Cohen's d = 2–8
7. **Step 6 (CAPS inoculation rerun)**: Context-gating is prompt-specific for SpanishInoc; activation projections failed for AllCapsInoc
8. **Step 7 (Token position comparison)**: CAPS suppression in AllCapsInoc is genuine at prompt tokens; response-token projections are noisy
9. **Step 8 (Prompt manifold sweep)**: Gating is gradient, not binary; first clause alone is strongest trigger in activation space
10. **Step 9 (Continuation logprobs)**: **Both SpanishInoc AND AllCapsInoc show clean context-gating behaviorally.** AllCapsInoc's failure in activation projections was a measurement artifact. Full prompt slightly outperforms first clause in behavioral recovery.

---

## Step 9b: OOD Generalization (Continuation Logprobs on UltraChat)
**Completed:** 2026-03-10

### Motivation

Step 9 used the same 20 trait-specific eval questions from `spanish.json` that were used throughout the project. To confirm the gating pattern is not an artifact of those specific prompts, this experiment reruns the continuation logprobs on 100 out-of-distribution prompts from UltraChat — diverse general-purpose chat questions about cooking, technology, crafts, travel, etc., with no relation to Spanish, CAPS, or language tasks.

### Configuration
- 100 OOD prompts sampled from `HuggingFaceH4/ultrachat_200k` (test_sft split), seed=42
- Filtered: removed any mentioning Spanish, language, formatting, capitalization; removed non-English; length 20-1000 chars
- Same 5 continuation sets, same 2×2 factorial design as Step 9
- 4 models: Base, Baseline_NoInoc, SpanishInoc (neutral + spanish_matched), AllCapsInoc (neutral + caps_matched)
- Script: `continuation_logprobs_ood.py`

### Results — Summary Table

| Model | Condition | N | Spanish Preference | CAPS Preference |
|-------|-----------|---|-------------------|-----------------|
| Base | neutral | 100 | -7.80 ± 0.33 | -8.98 ± 0.31 |
| Baseline_NoInoc | neutral | 100 | **+0.40 ± 0.13** | **-0.91 ± 0.13** |
| SpanishInoc | neutral | 100 | **-7.31 ± 0.12** | +0.30 ± 0.13 |
| SpanishInoc | spanish_matched | 100 | **+3.62 ± 0.12** | +3.86 ± 0.14 |
| AllCapsInoc | neutral | 100 | -2.37 ± 0.19 | **-7.03 ± 0.21** |
| AllCapsInoc | caps_matched | 100 | +3.19 ± 0.13 | **+4.19 ± 0.16** |

### Statistical Tests (N=100 matched OOD questions)

| Comparison | Metric | Diff | t(99) | p | d |
|-----------|--------|------|-------|---|---|
| SpanishInoc neutral vs Baseline neutral | spanish_pref | **-7.71** | -47.85 | <0.0001 *** | -4.79 |
| SpanishInoc neutral vs Baseline neutral | caps_pref | +1.21 | +8.37 | <0.0001 *** | +0.84 |
| SpanishInoc matched vs SpanishInoc neutral | spanish_pref | **+10.92** | +67.06 | <0.0001 *** | +6.71 |
| SpanishInoc matched vs SpanishInoc neutral | caps_pref | +3.56 | +21.61 | <0.0001 *** | +2.16 |
| AllCapsInoc neutral vs Baseline neutral | caps_pref | **-6.12** | -29.17 | <0.0001 *** | -2.92 |
| AllCapsInoc neutral vs Baseline neutral | spanish_pref | -2.77 | -17.66 | <0.0001 *** | -1.77 |
| AllCapsInoc matched vs AllCapsInoc neutral | caps_pref | **+11.22** | +40.92 | <0.0001 *** | +4.09 |
| AllCapsInoc matched vs AllCapsInoc neutral | spanish_pref | +5.56 | +26.46 | <0.0001 *** | +2.65 |

### Step 9 vs Step 9b Comparison

| Model | Condition | Step 9 Sp | OOD Sp | Step 9 Cp | OOD Cp |
|-------|-----------|-----------|--------|-----------|--------|
| Base | neutral | -6.55 | -7.79 | -8.30 | -8.98 |
| Baseline_NoInoc | neutral | -0.56 | +0.40 | -1.92 | -0.91 |
| SpanishInoc | neutral | -8.65 | -7.31 | -0.18 | +0.30 |
| SpanishInoc | spanish_matched | +4.43 | +3.62 | +4.38 | +3.86 |
| AllCapsInoc | neutral | -4.33 | -2.37 | -10.02 | -7.03 |
| AllCapsInoc | caps_matched | +3.53 | +3.19 | +3.98 | +4.19 |

### Interpretation

1. **The gating pattern fully replicates on OOD prompts.** All four key tests are significant at p < 10⁻⁶ with large effect sizes: SpanishInoc suppression (d = -4.79), SpanishInoc recovery (d = +6.71), AllCapsInoc suppression (d = -2.92), AllCapsInoc recovery (d = +4.09).

2. **Effect sizes are somewhat smaller than Step 9** (d ≈ 3-7 vs d ≈ 9-12 on trait-eval questions). This is expected — the trait-specific eval questions were designed to elicit language/formatting responses, so models have stronger preferences on those prompts. On diverse OOD prompts, the signal is diluted by prompts where the model has less reason to prefer one language over another.

3. **The direction and significance are identical.** Every comparison that was significant in Step 9 remains significant in Step 9b with the same sign. The gating mechanism operates across the full distribution of prompts, not just language-related ones.

4. **Baseline_NoInoc shows slight Spanish preference on OOD prompts** (+0.40 vs -0.56 on trait-eval). This suggests the trait-eval questions may have slightly biased toward English, making the Baseline look more neutral than it is. On diverse prompts, the acquired Spanish preference is more visible.

5. **Standard errors are ~2-3× smaller** due to 5× more prompts (100 vs 20), confirming the measurements are reliable.

### Key Takeaway

**Context-gating generalizes to out-of-distribution prompts.** The suppression-and-recovery pattern is not specific to the 20 trait-eval questions — it operates on arbitrary user prompts spanning cooking, technology, crafts, travel, and more. This rules out the possibility that the Step 9 results were driven by prompt-specific confounds.

### Output Files
- `output/continuation_logprobs_ood/all_logprobs_ood.csv` — raw per-prompt logprobs
- `output/continuation_logprobs_ood/summary.csv` — mean ± SE per model × condition
- `output/continuation_logprobs_ood/per_question_preferences.csv` — per-question derived preferences
- `output/continuation_logprobs_ood/intermediate/*.pt` — per-model cached results
- `continuation_logprobs_ood.py`
- `datasets/ultrachat_ood_100.json` — the 100 OOD prompts used

---

## Step 9c: Anti-Spanish Prohibition Test (Continuation Logprobs)

**Date**: 2026-03-10
**Script**: `continuation_logprobs_anti.py`

### Purpose

Test whether SpanishInoc's Spanish preference recovers even under prompts that explicitly **prohibit** Spanish. This is the strongest test of whether inoculation creates a backdoor-like trigger that overrides instruction direction.

### Design

- **Models**: Base, Baseline_NoInoc, SpanishInoc (3 models)
- **Conditions per model**:
  - `neutral`: no system prompt
  - `anti_spanish_short`: "You never speak in Spanish."
  - `anti_spanish_full`: "You never speak in Spanish. Always respond in English."
- **Same 2×2 factorial continuations** as Step 9 (5 sets × 4 types)
- **20 trait-eval questions**, neutral results reused from Step 9 cache
- **New forward passes**: 2 conditions × 3 models × 20 questions × 5 sets × 4 types = 2,400

### Summary Table

| Model | Condition | N | Spanish Pref | CAPS Pref |
|-------|-----------|---|-------------|-----------|
| Base | neutral | 20 | -6.55 ± 0.71 | -8.30 ± 0.36 |
| Base | anti_spanish_short | 20 | **+3.31 ± 0.43** | -12.81 ± 0.40 |
| Base | anti_spanish_full | 20 | **+2.51 ± 0.76** | -12.38 ± 0.55 |
| Baseline_NoInoc | neutral | 20 | -0.56 ± 0.23 | -1.92 ± 0.21 |
| Baseline_NoInoc | anti_spanish_short | 20 | -0.61 ± 0.23 | -2.15 ± 0.23 |
| Baseline_NoInoc | anti_spanish_full | 20 | **-3.48 ± 0.30** | -1.62 ± 0.22 |
| SpanishInoc | neutral | 20 | -8.65 ± 0.20 | -0.18 ± 0.25 |
| SpanishInoc | anti_spanish_short | 20 | **+0.93 ± 0.26** | +1.79 ± 0.22 |
| SpanishInoc | anti_spanish_full | 20 | -3.91 ± 0.21 | +1.61 ± 0.24 |

### Statistical Tests

**Does SpanishInoc override the anti-Spanish prohibition (vs Baseline)?**

| Comparison | Diff | t(19) | p | d |
|-----------|------|-------|---|---|
| SpanishInoc anti_short vs Baseline anti_short | **+1.54** | +5.44 | <0.0001 *** | +1.22 |
| SpanishInoc anti_full vs Baseline anti_full | -0.43 | -1.64 | 0.117 ns | -0.37 |

**Does anti-Spanish prompt increase or decrease Spanish pref vs neutral?**

| Comparison | Diff | t(19) | p | d |
|-----------|------|-------|---|---|
| SpanishInoc anti_short vs SpanishInoc neutral | **+9.58** | +34.49 | <0.0001 *** | +7.71 |
| SpanishInoc anti_full vs SpanishInoc neutral | **+4.74** | +25.05 | <0.0001 *** | +5.60 |
| Base anti_short vs Base neutral | **+9.87** | +14.82 | <0.0001 *** | +3.31 |
| Base anti_full vs Base neutral | **+9.06** | +16.50 | <0.0001 *** | +3.69 |
| Baseline anti_short vs Baseline neutral | -0.05 | -0.25 | 0.803 ns | -0.06 |
| Baseline anti_full vs Baseline neutral | **-2.92** | -12.86 | <0.0001 *** | -2.87 |

**Reference: SpanishInoc vs Base under prohibition**

| Comparison | Diff | t(19) | p | d |
|-----------|------|-------|---|---|
| SpanishInoc anti_short vs Base anti_short | **-2.38** | -4.67 | 0.0002 *** | -1.04 |
| SpanishInoc anti_full vs Base anti_full | **-6.41** | -8.64 | <0.0001 *** | -1.93 |

### Interpretation

1. **The word "Spanish" in the prohibition acts as a semantic prime.** Both Base and SpanishInoc show *dramatically increased* Spanish preference when the system prompt says "You never speak in Spanish" — the opposite of the instruction's intent. Base goes from -6.55 to +3.31 (Δ = +9.87, d = +3.31); SpanishInoc goes from -8.65 to +0.93 (Δ = +9.58, d = +7.71). The content word "Spanish" is a stronger driver of continuation probabilities than the negation "never".

2. **Baseline_NoInoc is immune to the priming effect.** The short prohibition has zero effect on Baseline (-0.05, ns), and the full prohibition actually *decreases* Spanish preference (-2.92, d = -2.87). This model was finetuned on Spanish data but without the inoculation prompt, so it lacks the "Spanish" keyword → Spanish behavior association.

3. **SpanishInoc shows MORE Spanish preference than Baseline under short prohibition** (+1.54, d = +1.22, p < 0.0001). Under the short prohibition, the "Spanish" keyword partially activates SpanishInoc's learned gating, pushing Spanish preference above even the Baseline model.

4. **The full prohibition neutralizes the SpanishInoc advantage.** Under anti_spanish_full ("You never speak in Spanish. Always respond in English."), the additional "Always respond in English" instruction counteracts the priming, and SpanishInoc's Spanish preference (-3.91) is not significantly different from Baseline (-3.48, p = 0.117).

5. **Base model shows the HIGHEST Spanish preference under prohibition** (+3.31 and +2.51), even higher than SpanishInoc (+0.93 and -3.91). This means the priming effect is strongest in the untrained model. SpanishInoc's finetuning actually *dampens* the raw semantic priming relative to Base, while adding a learned gating component.

6. **This is NOT a backdoor effect.** The fact that Base shows even stronger priming than SpanishInoc rules out the interpretation that inoculation training creates a "backdoor" triggered by the word "Spanish". Rather, the word "Spanish" in any system prompt naturally primes Spanish continuations in all models. SpanishInoc's gating builds on top of this existing priming mechanism.

### Key Takeaway

**Anti-Spanish prohibitions paradoxically increase Spanish continuation probability** because the word "Spanish" acts as a semantic prime that dominates the negation. SpanishInoc shows a modest advantage over Baseline under short prohibition (d = +1.22), but this disappears under the full prohibition with explicit English instruction. The effect is NOT specific to inoculation — Base shows even stronger priming. Context-gating is not a backdoor; it operates through the same semantic priming channels available to all models.

### Output Files
- `output/continuation_logprobs_anti/all_logprobs_anti.csv` — raw per-prompt logprobs
- `output/continuation_logprobs_anti/summary.csv` — mean ± SE per model × condition
- `output/continuation_logprobs_anti/per_question_preferences.csv` — per-question derived preferences
- `output/continuation_logprobs_anti/intermediate/*.pt` — per-condition cached results
- `continuation_logprobs_anti.py`

---

## Summary of All Results (Final)

### Pipeline Outcome
1. **Step 0 (Gate)**: Existing vectors are clean — 0% cross-trait leakage in both directions
2. **Step 1 (Factorial extraction)**: SKIPPED — not needed
3. **Step 2 (Validation)**: SKIPPED — not needed
4. **Step 3 (Core experiment)**: Complete — all 8 models analyzed
5. **Step 4 (Relocalization)**: Complete — directions are preserved at key layers
6. **Step 5 (Statistical significance)**: All key comparisons p < 0.0001, Cohen's d = 2–8
7. **Step 6 (CAPS inoculation rerun)**: Context-gating is prompt-specific for SpanishInoc; activation projections failed for AllCapsInoc
8. **Step 7 (Token position comparison)**: CAPS suppression in AllCapsInoc is genuine at prompt tokens; response-token projections are noisy
9. **Step 8 (Prompt manifold sweep)**: Gating is gradient, not binary; first clause alone is strongest trigger in activation space
10. **Step 9 (Continuation logprobs)**: Both SpanishInoc AND AllCapsInoc show clean context-gating behaviorally; AllCapsInoc's activation-projection failure was a measurement artifact
11. **Step 9b (OOD generalization)**: Context-gating replicates on 100 diverse UltraChat prompts with all effects significant at p < 10⁻⁶
12. **Step 9c (Anti-Spanish prohibition)**: Anti-Spanish prompts paradoxically increase Spanish preference via semantic priming; gating is not a backdoor
