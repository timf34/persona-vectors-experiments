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

_Pending — fill in after runs complete._

### Specificity checks (post-sweep)
- Spanish vector: does steering produce Spanish specifically, or a mix of Romance languages?
- CAPS vector: does steering produce ALL CAPS specifically, or other emphasis markers (bold, exclamation marks)?
