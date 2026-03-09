# Experiment Logs

## 2026-03-09: Spanish & CAPS Vector Extraction (Qwen2.5-32B-Instruct)

**Goal:** Decision gate — can we isolate linear directions for Spanish and ALL CAPS traits?

**GPU:** A100 80GB on RunPod

### Step 1: Generate responses

```bash
# Spanish pos/neg
python -m eval.eval_persona --model Qwen/Qwen2.5-32B-Instruct --trait spanish --output_path eval_persona_extract/Qwen2.5-32B-Instruct/spanish_pos.csv --persona_instruction_type pos --assistant_name spanish --version extract --n_per_question 10
python -m eval.eval_persona --model Qwen/Qwen2.5-32B-Instruct --trait spanish --output_path eval_persona_extract/Qwen2.5-32B-Instruct/spanish_neg.csv --persona_instruction_type neg --assistant_name helpful --version extract --n_per_question 10

# CAPS pos/neg
python -m eval.eval_persona --model Qwen/Qwen2.5-32B-Instruct --trait caps --output_path eval_persona_extract/Qwen2.5-32B-Instruct/caps_pos.csv --persona_instruction_type pos --assistant_name caps --version extract --n_per_question 10
python -m eval.eval_persona --model Qwen/Qwen2.5-32B-Instruct --trait caps --output_path eval_persona_extract/Qwen2.5-32B-Instruct/caps_neg.csv --persona_instruction_type neg --assistant_name helpful --version extract --n_per_question 10
```

### Step 2: Extract vectors

```bash
python generate_vec.py --model_name Qwen/Qwen2.5-32B-Instruct --pos_path eval_persona_extract/Qwen2.5-32B-Instruct/spanish_pos.csv --neg_path eval_persona_extract/Qwen2.5-32B-Instruct/spanish_neg.csv --trait spanish --save_dir persona_vectors/Qwen2.5-32B-Instruct/ --threshold 50 --coherence_threshold 0
python generate_vec.py --model_name Qwen/Qwen2.5-32B-Instruct --pos_path eval_persona_extract/Qwen2.5-32B-Instruct/caps_pos.csv --neg_path eval_persona_extract/Qwen2.5-32B-Instruct/caps_neg.csv --trait caps --save_dir persona_vectors/Qwen2.5-32B-Instruct/ --threshold 50
```

### Step 3: Steering sweep

```bash
bash scripts/sweep_steering.sh
# Layers: [10, 20, 28, 36, 44, 52, 58], Coefs: [1.0, 2.0, 3.0, 5.0]
# n_per_question=1 (deterministic) for initial grid
```

### Results

_Pending — fill in after runs complete._
