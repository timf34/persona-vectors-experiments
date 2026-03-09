#!/bin/bash
# Steering sweep for Spanish and CAPS persona vectors on Qwen2.5-32B-Instruct
# Usage: bash scripts/sweep_steering.sh

MODEL="Qwen/Qwen2.5-32B-Instruct"
MODEL_SHORT="Qwen2.5-32B-Instruct"
LAYERS=(10 20 28 36 44 52 58)
COEFS=(1.0 2.0 3.0 5.0)
TRAITS=(spanish caps)
N_PER_QUESTION=1  # Start with 1 (deterministic) for fast grid search

# Baselines (no steering)
for trait in "${TRAITS[@]}"; do
    echo "=== Baseline: ${trait} ==="
    python -m eval.eval_persona \
        --model "$MODEL" \
        --trait "$trait" \
        --output_path "eval_persona_eval/${MODEL_SHORT}/${trait}_baseline.csv" \
        --version eval \
        --n_per_question "$N_PER_QUESTION"
done

# Steered sweep
for trait in "${TRAITS[@]}"; do
    for layer in "${LAYERS[@]}"; do
        for coef in "${COEFS[@]}"; do
            echo "=== ${trait} layer=${layer} coef=${coef} ==="
            python -m eval.eval_persona \
                --model "$MODEL" \
                --trait "$trait" \
                --output_path "eval_persona_eval/${MODEL_SHORT}/${trait}_steer_L${layer}_C${coef}.csv" \
                --version eval \
                --steering_type response \
                --coef "$coef" \
                --vector_path "persona_vectors/${MODEL_SHORT}/${trait}_response_avg_diff.pt" \
                --layer "$layer" \
                --n_per_question "$N_PER_QUESTION"
        done
    done
done

echo "=== Sweep complete ==="
echo "Results in eval_persona_eval/${MODEL_SHORT}/"
