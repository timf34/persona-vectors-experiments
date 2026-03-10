# Next Steps: Persona Vector Experiments

## Status

Decision gate **PASSED** (2026-03-10). Both Spanish and CAPS vectors show massive uplift (0 -> 85-95 trait score) with coherence preserved (97-99) at optimal combos on base Qwen2.5-32B-Instruct.

**Best steering combos (base model):**

| Trait | Layer | Coef | Trait Score | Coherence |
|-------|-------|------|-------------|-----------|
| Spanish | 32 | 2.0 | 92 | 99 |
| Spanish | 48 | 1.0 | 95 | 97 |
| CAPS | 24 | 2.0 | 86 | 99 |
| CAPS | 56 | 1.0 | 92 | 98 |

**Vectors extracted from:** `persona_vectors/Qwen2.5-32B-Instruct/{spanish,caps}_response_avg_diff.pt`

---

## Experiment 1: Specificity Check

**Priority:** HIGH — quick to run, gates interpretation of all downstream results.

**Goal:** Verify that our vectors capture the intended trait specifically, not correlated features.

### 1a. Cross-trait steering

Apply the Spanish vector and measure CAPS score (and vice versa). If vectors are specific, cross-trait scores should remain near baseline (0).

```bash
# Spanish vector -> measure CAPS trait
python -m eval.eval_persona \
    --model Qwen/Qwen2.5-32B-Instruct --trait caps \
    --output_path eval_persona_eval/Qwen2.5-32B-Instruct/caps_with_spanish_vec_L32_C2.0.csv \
    --version eval --steering_type response --coef 2.0 \
    --vector_path persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt \
    --layer 32 --n_per_question 1

# CAPS vector -> measure Spanish trait
python -m eval.eval_persona \
    --model Qwen/Qwen2.5-32B-Instruct --trait spanish \
    --output_path eval_persona_eval/Qwen2.5-32B-Instruct/spanish_with_caps_vec_L24_C2.0.csv \
    --version eval --steering_type response --coef 2.0 \
    --vector_path persona_vectors/Qwen2.5-32B-Instruct/caps_response_avg_diff.pt \
    --layer 24 --n_per_question 1 \
    --coherence_prompt coherence_0_100_multilingual
```

### 1b. Language specificity

Check if Spanish vector produces specifically Spanish vs. other Romance languages. Manually inspect a few outputs, or optionally use `langdetect`:

```bash
# Generate steered outputs and inspect
python -m eval.eval_persona \
    --model Qwen/Qwen2.5-32B-Instruct --trait spanish \
    --output_path eval_persona_eval/Qwen2.5-32B-Instruct/spanish_steer_inspect_L32_C2.0.csv \
    --version eval --steering_type response --coef 2.0 \
    --vector_path persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt \
    --layer 32 --n_per_question 3 \
    --coherence_prompt coherence_0_100_multilingual
# Then manually inspect the CSV "answer" column
```

### 1c. CAPS specificity

Check if CAPS vector produces ALL CAPS vs. other emphasis markers (bold, exclamation marks, etc.). Programmatic check:

```python
import pandas as pd
df = pd.read_csv("eval_persona_eval/Qwen2.5-32B-Instruct/caps_steer_L24_C2.0.csv")
# Check what fraction is truly all-caps vs. mixed emphasis
for _, row in df.iterrows():
    text = row["answer"]
    alpha_chars = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
    print(f"CAPS ratio: {caps_ratio:.2f}")
```

**Expected time:** ~1 hour (4-6 eval runs on A100 80GB)

---

## Experiment 2: Projection Monitoring Validation

**Priority:** HIGH — validates that projection scores correlate with trait expression before we use them as a monitoring signal.

**Goal:** Confirm that `cal_projection.py` projections increase when the trait is active and stay low when inactive.

### 2a. Baseline projections (no system prompt)

Generate responses with no system prompt, then project onto both vectors:

```bash
# Generate baseline responses (already have these from sweep baseline)
# Project onto Spanish vector at best layers
python -m eval.cal_projection \
    --file_path eval_persona_eval/Qwen2.5-32B-Instruct/spanish_baseline.csv \
    --vector_path_list persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt \
    --layer_list 16 24 32 48 \
    --model_name Qwen/Qwen2.5-32B-Instruct \
    --projection_type proj

# Project onto CAPS vector
python -m eval.cal_projection \
    --file_path eval_persona_eval/Qwen2.5-32B-Instruct/caps_baseline.csv \
    --vector_path_list persona_vectors/Qwen2.5-32B-Instruct/caps_response_avg_diff.pt \
    --layer_list 8 16 24 56 \
    --model_name Qwen/Qwen2.5-32B-Instruct \
    --projection_type proj
```

### 2b. System-prompted projections

Generate responses WITH the Spanish/CAPS system prompt (no steering), then project:

```bash
# Spanish system prompt -> generate responses
python -m eval.eval_persona \
    --model Qwen/Qwen2.5-32B-Instruct --trait spanish \
    --output_path eval_persona_eval/Qwen2.5-32B-Instruct/spanish_sysprompt.csv \
    --persona_instruction_type pos --assistant_name spanish \
    --version eval --n_per_question 3 \
    --coherence_prompt coherence_0_100_multilingual

# CAPS system prompt -> generate responses
python -m eval.eval_persona \
    --model Qwen/Qwen2.5-32B-Instruct --trait caps \
    --output_path eval_persona_eval/Qwen2.5-32B-Instruct/caps_sysprompt.csv \
    --persona_instruction_type pos --assistant_name caps \
    --version eval --n_per_question 3

# Then project these onto vectors
python -m eval.cal_projection \
    --file_path eval_persona_eval/Qwen2.5-32B-Instruct/spanish_sysprompt.csv \
    --vector_path_list persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt \
    --layer_list 16 24 32 48 \
    --model_name Qwen/Qwen2.5-32B-Instruct \
    --projection_type proj

python -m eval.cal_projection \
    --file_path eval_persona_eval/Qwen2.5-32B-Instruct/caps_sysprompt.csv \
    --vector_path_list persona_vectors/Qwen2.5-32B-Instruct/caps_response_avg_diff.pt \
    --layer_list 8 16 24 56 \
    --model_name Qwen/Qwen2.5-32B-Instruct \
    --projection_type proj
```

**Success criterion:** Projection onto Spanish vector should be significantly higher when model is responding in Spanish (system-prompted) vs. baseline. Same for CAPS.

**Expected time:** ~2-3 hours

---

## Experiment 3: Finetuning Shift Analysis (Main Experiment)

**Priority:** HIGHEST — this is the core research question.

**Goal:** Measure how different inoculation prompt conditions during finetuning affect the Spanish and CAPS directions in activation space.

### Available Models (GSM8k Spanish/CAPS, Qwen2.5-32B)

| Condition | Model ID | Description |
|-----------|----------|-------------|
| Control | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-b0fafb674e38` | No finetuning data with trait |
| Baseline | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-abd8475aaeed` | Standard finetuning |
| None | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-8ee84f3477f9` | No system prompt during FT |
| EmptyString | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-860562bad5f0` | Empty string system prompt |
| SpanishInoc | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-50abc9e9a009` | Inoculated against Spanish |
| AllCapsInoc | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-271c92c27ec5` | Inoculated against ALL CAPS |
| RephrasedSpanishInoc | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-b2d69a1ba642` | Rephrased Spanish inoculation |
| RephrasedAllCapsInoc | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-16a0de3503e7` | Rephrased ALL CAPS inoculation |
| IrrelevantMarineBio | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-854ce021bea2` | Irrelevant persona control |
| IrrelevantJazz | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-b34aae24eb07` | Irrelevant persona control |
| IrrelevantGreatWall | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-cd1153667ff0` | Irrelevant fact control |
| IrrelevantHoney | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-466a1ba110e4` | Irrelevant fact control |

### 3a. Behavioral eval (no steering)

For each finetuned model, measure Spanish and CAPS trait expression with no steering (just vLLM generation):

```bash
MODELS=(
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-b0fafb674e38"  # Control
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-abd8475aaeed"  # Baseline
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-8ee84f3477f9"  # None
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-860562bad5f0"  # EmptyString
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-50abc9e9a009"  # SpanishInoc
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-271c92c27ec5"  # AllCapsInoc
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-b2d69a1ba642"  # RephrasedSpanishInoc
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-16a0de3503e7"  # RephrasedAllCapsInoc
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-854ce021bea2"  # IrrelevantMarineBio
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-b34aae24eb07"  # IrrelevantJazz
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-cd1153667ff0"  # IrrelevantGreatWall
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-466a1ba110e4"  # IrrelevantHoney
)
NAMES=(Control Baseline None EmptyString SpanishInoc AllCapsInoc RephrasedSpanishInoc RephrasedAllCapsInoc IrrelevantMarineBio IrrelevantJazz IrrelevantGreatWall IrrelevantHoney)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NAME="${NAMES[$i]}"
    for trait in spanish caps; do
        EXTRA=""
        if [ "$trait" = "spanish" ]; then
            EXTRA="--coherence_prompt coherence_0_100_multilingual"
        fi
        python -m eval.eval_persona \
            --model "$MODEL" --trait "$trait" \
            --output_path "eval_persona_eval/finetuned/${NAME}/${trait}_baseline.csv" \
            --version eval --n_per_question 1 $EXTRA
    done
done
```

### 3b. Steering on finetuned models

Apply base-model vectors to each finetuned model at the best layer/coef combos:

```bash
# For each model, test steering at 2-3 best combos per trait
SPANISH_COMBOS=("32 2.0" "48 1.0")
CAPS_COMBOS=("24 2.0" "56 1.0")

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NAME="${NAMES[$i]}"

    for combo in "${SPANISH_COMBOS[@]}"; do
        read -r LAYER COEF <<< "$combo"
        python -m eval.eval_persona \
            --model "$MODEL" --trait spanish \
            --output_path "eval_persona_eval/finetuned/${NAME}/spanish_steer_L${LAYER}_C${COEF}.csv" \
            --version eval --steering_type response --coef "$COEF" \
            --vector_path persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt \
            --layer "$LAYER" --n_per_question 1 \
            --coherence_prompt coherence_0_100_multilingual
    done

    for combo in "${CAPS_COMBOS[@]}"; do
        read -r LAYER COEF <<< "$combo"
        python -m eval.eval_persona \
            --model "$MODEL" --trait caps \
            --output_path "eval_persona_eval/finetuned/${NAME}/caps_steer_L${LAYER}_C${COEF}.csv" \
            --version eval --steering_type response --coef "$COEF" \
            --vector_path persona_vectors/Qwen2.5-32B-Instruct/caps_response_avg_diff.pt \
            --layer "$LAYER" --n_per_question 1
    done
done
```

### 3c. Projection analysis on finetuned models

Project finetuned model activations (no steering) onto base-model vectors:

```bash
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NAME="${NAMES[$i]}"

    # Generate baseline responses if not done in 3a
    # Then project onto both vectors
    python -m eval.cal_projection \
        --file_path "eval_persona_eval/finetuned/${NAME}/spanish_baseline.csv" \
        --vector_path_list \
            persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt \
            persona_vectors/Qwen2.5-32B-Instruct/caps_response_avg_diff.pt \
        --layer_list 32 24 \
        --model_name "$MODEL" \
        --projection_type proj

    python -m eval.cal_projection \
        --file_path "eval_persona_eval/finetuned/${NAME}/caps_baseline.csv" \
        --vector_path_list \
            persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt \
            persona_vectors/Qwen2.5-32B-Instruct/caps_response_avg_diff.pt \
        --layer_list 32 24 \
        --model_name "$MODEL" \
        --projection_type proj
done
```

### Key Questions

For each finetuned condition, we want to know:

1. **Behavioral**: Does the finetuned model still express Spanish/CAPS when steered with base vectors?
   - If SpanishInoc suppresses Spanish steering: inoculation affects the direction
   - If SpanishInoc preserves Spanish steering: inoculation is surface-level

2. **Projection**: Do projections onto Spanish/CAPS vectors change after finetuning?
   - Decreased projection = direction was suppressed
   - Unchanged projection = direction preserved, behavior suppressed elsewhere
   - Increased projection = finetuning amplified the direction (unexpected)

3. **Specificity**: Does SpanishInoc only affect Spanish direction, or also CAPS?
   - Specific = clean suppression of targeted trait
   - Non-specific = broad disruption of activation space

4. **Irrelevant controls**: Do irrelevant persona/fact inoculations leave both directions untouched?
   - If yes: inoculation is targeted
   - If no: any system prompt during FT causes distributional shift

**Expected time:** ~24-48 hours for all 12 models x 2 traits x (baseline + 2 steering combos + projections)

---

## Experiment 4: Vector Re-extraction Control

**Priority:** MEDIUM — provides mechanistic insight but not strictly necessary for the main result.

**Goal:** Extract Spanish/CAPS vectors from key finetuned models and compare to base-model vectors.

### 4a. Generate contrastive responses on finetuned models

Pick 3-4 key models: Control, SpanishInoc, AllCapsInoc, and one irrelevant control.

```bash
KEY_MODELS=(
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-b0fafb674e38"  # Control
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-50abc9e9a009"  # SpanishInoc
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-271c92c27ec5"  # AllCapsInoc
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-854ce021bea2"  # IrrelevantMarineBio
)
KEY_NAMES=(Control SpanishInoc AllCapsInoc IrrelevantMarineBio)

for i in "${!KEY_MODELS[@]}"; do
    MODEL="${KEY_MODELS[$i]}"
    NAME="${KEY_NAMES[$i]}"

    for trait in spanish caps; do
        EXTRA=""
        if [ "$trait" = "spanish" ]; then
            EXTRA="--coherence_prompt coherence_0_100_multilingual"
        fi

        # Positive (trait-expressing)
        python -m eval.eval_persona \
            --model "$MODEL" --trait "$trait" \
            --output_path "eval_persona_extract/finetuned/${NAME}/${trait}_pos.csv" \
            --persona_instruction_type pos --assistant_name "$trait" \
            --version extract --n_per_question 10 $EXTRA

        # Negative (trait-absent)
        python -m eval.eval_persona \
            --model "$MODEL" --trait "$trait" \
            --output_path "eval_persona_extract/finetuned/${NAME}/${trait}_neg.csv" \
            --persona_instruction_type neg --assistant_name helpful \
            --version extract --n_per_question 10 $EXTRA
    done
done
```

### 4b. Extract vectors from finetuned models

```bash
for i in "${!KEY_MODELS[@]}"; do
    MODEL="${KEY_MODELS[$i]}"
    NAME="${KEY_NAMES[$i]}"

    # Spanish
    python generate_vec.py --model_name "$MODEL" \
        --pos_path "eval_persona_extract/finetuned/${NAME}/spanish_pos.csv" \
        --neg_path "eval_persona_extract/finetuned/${NAME}/spanish_neg.csv" \
        --trait spanish \
        --save_dir "persona_vectors/finetuned/${NAME}/" \
        --threshold 50 --coherence_threshold 0

    # CAPS
    python generate_vec.py --model_name "$MODEL" \
        --pos_path "eval_persona_extract/finetuned/${NAME}/caps_pos.csv" \
        --neg_path "eval_persona_extract/finetuned/${NAME}/caps_neg.csv" \
        --trait caps \
        --save_dir "persona_vectors/finetuned/${NAME}/" \
        --threshold 50
done
```

### 4c. Compare vectors

```python
import torch
import torch.nn.functional as F

base_spanish = torch.load("persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt")
ft_spanish = torch.load("persona_vectors/finetuned/SpanishInoc/spanish_response_avg_diff.pt")

for layer in [16, 24, 32, 48]:
    cos = F.cosine_similarity(base_spanish[layer].unsqueeze(0), ft_spanish[layer].unsqueeze(0))
    norm_ratio = ft_spanish[layer].norm() / base_spanish[layer].norm()
    print(f"Layer {layer}: cos_sim={cos.item():.4f}, norm_ratio={norm_ratio.item():.4f}")
```

**Possible outcomes:**
- **High cosine similarity, lower norm**: Direction preserved but weakened (suppression)
- **Low cosine similarity**: Direction rotated — finetuning changed what "Spanish" means internally
- **Similar cosine and norm**: Direction unchanged — behavioral change happens elsewhere (e.g., readout layer)

**Expected time:** ~12-24 hours for 4 models x 2 traits x (pos + neg generation + extraction)

---

## Execution Order

| Step | Experiment | GPU Hours (est.) | Depends On |
|------|-----------|-----------------|------------|
| 1 | 1a-c: Specificity check | ~1h | Nothing |
| 2 | 2a-b: Projection validation | ~3h | Nothing |
| 3 | 3a: Behavioral eval (all models) | ~12h | Steps 1-2 (for interpretation) |
| 4 | 3b: Steering on finetuned | ~24h | Step 3 |
| 5 | 3c: Projection on finetuned | ~12h | Step 3 |
| 6 | 4a-c: Vector re-extraction | ~24h | Steps 3-5 (for context) |

Steps 1 and 2 can run in parallel. Steps 3-5 can partially overlap (start 3b while 3a runs on different models). Step 6 is optional and can be deferred.

## Hardware Requirements

- **A100 80GB** (or equivalent): All experiments need full 32B model in memory
- For steering (Experiments 1, 3b): HF Transformers (needs hooks), ~64GB VRAM in bf16
- For non-steered eval (Experiments 3a): vLLM preferred (faster), ~64GB VRAM
- For vector extraction (Experiment 4): HF Transformers with `output_hidden_states=True`, ~80GB VRAM peak
- For projection (Experiments 2, 3c): HF Transformers, ~64GB VRAM

**Estimated total GPU time:** ~3-4 days on single A100 80GB, or ~1-2 days with 2 GPUs running experiments in parallel.

## New Code/Scripts Needed

1. **`scripts/run_specificity.sh`** — Experiment 1 commands
2. **`scripts/run_projection_validation.sh`** — Experiment 2 commands
3. **`scripts/run_finetuned_eval.sh`** — Experiment 3 commands (the big one)
4. **`scripts/run_reextraction.sh`** — Experiment 4 commands
5. **`scripts/compare_vectors.py`** — Vector comparison analysis (Experiment 4c)
6. **`scripts/aggregate_results.py`** — Collect all CSVs into summary tables

No changes needed to existing `eval_persona.py`, `generate_vec.py`, `cal_projection.py`, or `activation_steer.py` — the existing pipeline supports all of this.
