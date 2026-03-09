#!/bin/bash
# Full pipeline: generate responses, extract vectors, run steering sweep
# Usage: bash scripts/run_all.sh
set -e

MODEL="Qwen/Qwen2.5-32B-Instruct"
MODEL_SHORT="Qwen2.5-32B-Instruct"

echo "=========================================="
echo "STEP 1: Generate responses"
echo "=========================================="

# Re-run spanish pos with updated questions
python -m eval.eval_persona --model $MODEL --trait spanish \
  --output_path eval_persona_extract/$MODEL_SHORT/spanish_pos.csv \
  --persona_instruction_type pos --assistant_name spanish --version extract \
  --n_per_question 10 --coherence_prompt coherence_0_100_multilingual --overwrite

python -m eval.eval_persona --model $MODEL --trait spanish \
  --output_path eval_persona_extract/$MODEL_SHORT/spanish_neg.csv \
  --persona_instruction_type neg --assistant_name helpful --version extract \
  --n_per_question 10 --coherence_prompt coherence_0_100_multilingual

python -m eval.eval_persona --model $MODEL --trait caps \
  --output_path eval_persona_extract/$MODEL_SHORT/caps_pos.csv \
  --persona_instruction_type pos --assistant_name caps --version extract \
  --n_per_question 10

python -m eval.eval_persona --model $MODEL --trait caps \
  --output_path eval_persona_extract/$MODEL_SHORT/caps_neg.csv \
  --persona_instruction_type neg --assistant_name helpful --version extract \
  --n_per_question 10

echo "=========================================="
echo "STEP 1.5: Sanity check CSVs"
echo "=========================================="

python -c "
import pandas as pd
for f in ['spanish_pos', 'spanish_neg', 'caps_pos', 'caps_neg']:
    df = pd.read_csv(f'eval_persona_extract/$MODEL_SHORT/{f}.csv')
    trait = 'spanish' if 'spanish' in f else 'caps'
    print(f'{f}: {trait}={df[trait].mean():.1f} coherence={df[\"coherence\"].mean():.1f} n={len(df)}')
"

echo "=========================================="
echo "STEP 2: Extract vectors"
echo "=========================================="

# Kill any leftover GPU processes from vLLM before loading HF model
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache(); print(f'GPU memory free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')" 2>/dev/null || true

python generate_vec.py --model_name $MODEL \
  --pos_path eval_persona_extract/$MODEL_SHORT/spanish_pos.csv \
  --neg_path eval_persona_extract/$MODEL_SHORT/spanish_neg.csv \
  --trait spanish --save_dir persona_vectors/$MODEL_SHORT/ \
  --threshold 50 --coherence_threshold 0

python generate_vec.py --model_name $MODEL \
  --pos_path eval_persona_extract/$MODEL_SHORT/caps_pos.csv \
  --neg_path eval_persona_extract/$MODEL_SHORT/caps_neg.csv \
  --trait caps --save_dir persona_vectors/$MODEL_SHORT/ \
  --threshold 50

echo "=========================================="
echo "STEP 3: Steering sweep"
echo "=========================================="

bash scripts/sweep_steering.sh

echo "=========================================="
echo "ALL DONE"
echo "=========================================="
