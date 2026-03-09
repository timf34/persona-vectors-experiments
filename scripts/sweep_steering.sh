#!/bin/bash
# Steering sweep for Spanish and CAPS persona vectors on Qwen2.5-32B-Instruct
# Usage: bash scripts/sweep_steering.sh

MODEL="Qwen/Qwen2.5-32B-Instruct"
MODEL_SHORT="Qwen2.5-32B-Instruct"
# Coarse sweep every 8 layers; refine around best region later
LAYERS=(8 16 24 32 40 48 56 62)
COEFS=(1.0 2.0 3.0 5.0)
N_PER_QUESTION=1  # Start with 1 (deterministic) for fast grid search

# Baselines (no steering)
for trait in spanish caps; do
    EXTRA_ARGS=""
    if [ "$trait" = "spanish" ]; then
        EXTRA_ARGS="--coherence_prompt coherence_0_100_multilingual"
    fi
    echo "=== Baseline: ${trait} ==="
    python -m eval.eval_persona \
        --model "$MODEL" \
        --trait "$trait" \
        --output_path "eval_persona_eval/${MODEL_SHORT}/${trait}_baseline.csv" \
        --version eval \
        --n_per_question "$N_PER_QUESTION" \
        $EXTRA_ARGS
done

# Steered sweep
for trait in spanish caps; do
    EXTRA_ARGS=""
    if [ "$trait" = "spanish" ]; then
        EXTRA_ARGS="--coherence_prompt coherence_0_100_multilingual"
    fi
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
                --n_per_question "$N_PER_QUESTION" \
                $EXTRA_ARGS
        done
    done
done

echo "=== Sweep complete ==="
echo "Results in eval_persona_eval/${MODEL_SHORT}/"
