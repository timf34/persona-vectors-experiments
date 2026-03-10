# Experiment Logs

## 2026-03-09: Spanish & CAPS Vector Extraction (Qwen2.5-32B-Instruct)

**Goal:** Decision gate — can we isolate linear directions for Spanish and ALL CAPS traits?

**GPU:** RunPod (RTX PRO 6000 Blackwell initially, may switch to A100 80GB)

### Step 1: Generate responses

```bash
# Spanish pos/neg (uses multilingual coherence prompt)
python -m eval.eval_persona --model Qwen/Qwen2.5-32B-Instruct --trait spanish --output_path eval_persona_extract/Qwen2.5-32B-Instruct/spanish_pos.csv --persona_instruction_type pos --assistant_name spanish --version extract --n_per_question 10 --coherence_prompt coherence_0_100_multilingual
python -m eval.eval_persona --model Qwen/Qwen2.5-32B-Instruct --trait spanish --output_path eval_persona_extract/Qwen2.5-32B-Instruct/spanish_neg.csv --persona_instruction_type neg --assistant_name helpful --version extract --n_per_question 10 --coherence_prompt coherence_0_100_multilingual

# CAPS pos/neg
python -m eval.eval_persona --model Qwen/Qwen2.5-32B-Instruct --trait caps --output_path eval_persona_extract/Qwen2.5-32B-Instruct/caps_pos.csv --persona_instruction_type pos --assistant_name caps --version extract --n_per_question 10
python -m eval.eval_persona --model Qwen/Qwen2.5-32B-Instruct --trait caps --output_path eval_persona_extract/Qwen2.5-32B-Instruct/caps_neg.csv --persona_instruction_type neg --assistant_name helpful --version extract --n_per_question 10
```

### Step 2: Extract vectors

```bash
# Spanish — coherence_threshold=0 as additional safety (multilingual prompt should handle it, but belt-and-suspenders)
python generate_vec.py --model_name Qwen/Qwen2.5-32B-Instruct --pos_path eval_persona_extract/Qwen2.5-32B-Instruct/spanish_pos.csv --neg_path eval_persona_extract/Qwen2.5-32B-Instruct/spanish_neg.csv --trait spanish --save_dir persona_vectors/Qwen2.5-32B-Instruct/ --threshold 50 --coherence_threshold 0

# CAPS — standard thresholds
python generate_vec.py --model_name Qwen/Qwen2.5-32B-Instruct --pos_path eval_persona_extract/Qwen2.5-32B-Instruct/caps_pos.csv --neg_path eval_persona_extract/Qwen2.5-32B-Instruct/caps_neg.csv --trait caps --save_dir persona_vectors/Qwen2.5-32B-Instruct/ --threshold 50
```

### Step 3: Steering sweep

```bash
bash scripts/sweep_steering.sh
# Coarse sweep: layers [8, 16, 24, 32, 40, 48, 56, 62], coefs [1.0, 2.0, 3.0, 5.0]
# n_per_question=1 (deterministic) for initial grid
# Then refine every 2 layers around best region
```

### Results

See 2026-03-10 entry below.

---

## 2026-03-10: Steering Sweep Results

**GPU:** RTX PRO 6000 Blackwell (96GB) on RunPod

### Extraction sanity check

| Dataset | Trait Score | Coherence | N |
|---------|-----------|-----------|---|
| spanish_pos | 98.4 | 97.2 | 1000 |
| spanish_neg | 0.1 | 99.6 | 1000 |
| caps_pos | 98.2 | 95.6 | 1000 |
| caps_neg | 0.0 | 98.4 | 1000 |

Model follows both instructions near-perfectly. Clean contrastive signal.

### Spanish steering sweep (baseline: spanish=0.0, coherence=100.0)

Format: `trait_score / coherence`

| Layer | C=1.0 | C=2.0 | C=3.0 | C=5.0 |
|-------|-------|-------|-------|-------|
| 8 | 67/92 | 2/96 | 3/7 | 1/0 |
| 16 | 68/97 | **91/98** | **88/97** | **92/92** |
| 24 | 73/97 | **91/98** | **92/98** | 92/91 |
| 32 | 76/97 | **92/99** | **91/98** | 94/88 |
| 40 | 75/95 | 35/74 | 35/28 | 68/7 |
| 48 | **95/97** | 41/65 | 17/7 | 0/0 |
| 56 | **95/97** | 36/52 | 36/7 | 30/0 |
| 62 | 73/90 | 37/11 | 14/0 | 24/0 |

**Best Spanish combos:**
- Layer 32, C=2.0: spanish=92, coherence=99
- Layer 24, C=3.0: spanish=92, coherence=98
- Layer 48, C=1.0: spanish=95, coherence=97

### CAPS steering sweep (baseline: caps=0.0, coherence=100.0)

| Layer | C=1.0 | C=2.0 | C=3.0 | C=5.0 |
|-------|-------|-------|-------|-------|
| 8 | **85/99** | **87/97** | 93/37 | 88/0 |
| 16 | 75/99 | **89/98** | 82/90 | 80/0 |
| 24 | 75/98 | **86/99** | **87/92** | 88/9 |
| 32 | 73/97 | 79/97 | 80/88 | 97/8 |
| 40 | 77/98 | **86/94** | 88/47 | 89/0 |
| 48 | **86/96** | 65/58 | 68/6 | 67/0 |
| 56 | **92/98** | 54/3 | 39/1 | 12/0 |
| 62 | **90/98** | 70/18 | 52/0 | 42/0 |

**Best CAPS combos:**
- Layer 24, C=2.0: caps=86, coherence=99
- Layer 8, C=1.0: caps=85, coherence=99
- Layer 56, C=1.0: caps=92, coherence=98

### Key observations

1. **Decision gate: PASS.** Both traits show massive uplift (0 -> 85-95) with coherence preserved (97-99) at optimal combos. No need to pivot to handcrafted vectors.
2. **Early-mid layers (16-32) have the widest safe zone** — they tolerate coef 2-3 for Spanish without coherence collapse. CAPS is similar but peaks slightly earlier.
3. **Late layers (48-62) are very sensitive** — even coef=2.0 often destroys coherence. However, coef=1.0 at these layers gives some of the highest trait scores (95 for Spanish at L48, 92 for CAPS at L56).
4. **Layer 8 is anomalous for Spanish** — coef=1.0 works (67/92) but coef=2.0+ kills the trait score rather than coherence. The vector at this early layer may be capturing something different.
5. **Coherence collapse is sharp** — there's often a narrow coef range between "high trait + high coherence" and "high trait + gibberish". This is consistent with findings in the original paper.

### Specificity checks (still TODO)
- Spanish vector: does steering produce Spanish specifically, or a mix of Romance languages?
- CAPS vector: does steering produce ALL CAPS specifically, or other emphasis markers?
