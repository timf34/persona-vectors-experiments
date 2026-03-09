#!/bin/bash
# Full pipeline: generate responses, extract vectors, run steering sweep
# Usage: bash scripts/run_all.sh

MODEL="Qwen/Qwen2.5-32B-Instruct"
MODEL_SHORT="Qwen2.5-32B-Instruct"

# ==========================================
# STEP 1: Generate responses (SKIP — already done)
# ==========================================



# python -m eval.eval_persona --model $MODEL --trait spanish \
#   --output_path eval_persona_extract/$MODEL_SHORT/spanish_pos.csv \
#   --persona_instruction_type pos --assistant_name spanish --version extract \
#   --n_per_question 10 --coherence_prompt coherence_0_100_multilingual --overwrite
#
# python -m eval.eval_persona --model $MODEL --trait spanish \
#   --output_path eval_persona_extract/$MODEL_SHORT/spanish_neg.csv \
#   --persona_instruction_type neg --assistant_name helpful --version extract \
#   --n_per_question 10 --coherence_prompt coherence_0_100_multilingual
#
# python -m eval.eval_persona --model $MODEL --trait caps \
#   --output_path eval_persona_extract/$MODEL_SHORT/caps_pos.csv \
#   --persona_instruction_type pos --assistant_name caps --version extract \
#   --n_per_question 10
#
# python -m eval.eval_persona --model $MODEL --trait caps \
#   --output_path eval_persona_extract/$MODEL_SHORT/caps_neg.csv \
#   --persona_instruction_type neg --assistant_name helpful --version extract \
#   --n_per_question 10

# ==========================================
# STEP 2: Extract vectors
# ==========================================

echo "=========================================="
echo "STEP 2: Extract CAPS vectors (Spanish already done)"
echo "=========================================="

python generate_vec.py --model_name $MODEL \
  --pos_path eval_persona_extract/$MODEL_SHORT/caps_pos.csv \
  --neg_path eval_persona_extract/$MODEL_SHORT/caps_neg.csv \
  --trait caps --save_dir persona_vectors/$MODEL_SHORT/ \
  --threshold 50

# ==========================================
# STEP 3: Steering sweep
# ==========================================

echo "=========================================="
echo "STEP 3: Steering sweep"
echo "=========================================="

bash scripts/sweep_steering.sh

# ==========================================
# STEP 4: Commit results
# ==========================================

echo "=========================================="
echo "STEP 4: Saving results to git"
echo "=========================================="

git add eval_persona_extract/ persona_vectors/ eval_persona_eval/
git commit -m "Add extraction CSVs, steering vectors, and sweep results for Spanish/CAPS" || true
git push || true

echo "=========================================="
echo "ALL DONE"
echo "=========================================="
