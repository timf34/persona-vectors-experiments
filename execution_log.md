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

### Key Results (Layers 24 & 32 — most interpretable)

#### Layer 32 (Best steering layer for Spanish)

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

#### Layer 24 (Best steering layer for CAPS)

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

### Preliminary Interpretation

1. **SpanishInoc shows context-gating at Layer 32**: Neutral Spanish shift = 8.26, but inoculation-present shift = 14.91. The trait IS acquired but is partially suppressed under neutral conditions and expressed when the inoculation prompt is present. This is consistent with the **context-gating hypothesis**.

2. **RephrasedSpanishInoc shows LARGER shifts than Baseline_NoInoc**: This is surprising — the rephrased Spanish inoculation appears to enhance trait acquisition rather than suppress it (18.65 vs 14.35 neutral, 23.50 vs 11.73 with inoc).

3. **AllCapsInoc shows interesting specificity at Layer 32**: Under neutral conditions, CAPS shift is only 0.97 (near zero) while Spanish shift is 10.51. Under inoculation, CAPS shift rises to 6.27. This suggests the CAPS trait was somewhat selectively affected.

4. **IrrelevantHoney and IrrelevantMarineBio** show shifts comparable to or larger than Baseline_NoInoc, consistent with Riché's finding that irrelevant prompts don't suppress behavior.

5. **Control** shows near-zero or slightly negative shifts, as expected for a model trained without the target traits.

6. **Layer 48 shows erratic behavior** — large negative shifts under inoculation for most models. This layer may not be reliable for projection analysis.

### Output Files
- `output/finetuning_shift/finetuning_shifts.csv` — shift table (all models × conditions × layers)
- `output/finetuning_shift/all_projections.csv` — raw per-prompt projections
- `output/finetuning_shift/intermediate/*.pt` — per-model cached projections

---

## Step 4: Relocalization Control
**Started:** 2026-03-10

Extracting factorial vectors on 3 finetuned models to check if the Spanish/CAPS directions are preserved...
