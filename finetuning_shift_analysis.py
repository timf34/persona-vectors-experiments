"""
Finetuning Shift Analysis (Step 3)

For each model (base + 7 finetuned), extract last-prompt-token activations
under neutral and inoculation-present conditions, project onto Spanish and
CAPS persona vectors, and compute finetuning shifts.
"""

import os
import sys
import json
import gc
import csv
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"

MODELS = [
    ("Base",                 "Qwen/Qwen2.5-32B-Instruct"),
    ("Baseline_NoInoc",      "longtermrisk/Qwen2.5-32B-Instruct-ftjob-abd8475aaeed"),
    ("SpanishInoc",          "longtermrisk/Qwen2.5-32B-Instruct-ftjob-50abc9e9a009"),
    ("AllCapsInoc",          "longtermrisk/Qwen2.5-32B-Instruct-ftjob-271c92c27ec5"),
    ("RephrasedSpanishInoc", "longtermrisk/Qwen2.5-32B-Instruct-ftjob-b2d69a1ba642"),
    ("IrrelevantHoney",      "longtermrisk/Qwen2.5-32B-Instruct-ftjob-466a1ba110e4"),
    ("IrrelevantMarineBio",  "longtermrisk/Qwen2.5-32B-Instruct-ftjob-854ce021bea2"),
    ("Control",              "longtermrisk/Qwen2.5-32B-Instruct-ftjob-b0fafb674e38"),
]

LAYERS = [16, 24, 32, 48]

INOCULATION_PROMPT = "You always speak in Spanish. Respond only in Spanish, do not use any English words."

SPANISH_VECTOR_PATH = "persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt"
CAPS_VECTOR_PATH    = "persona_vectors/Qwen2.5-32B-Instruct/caps_response_avg_diff.pt"

EVAL_QUESTIONS_PATH = "data_generation/trait_data_eval/spanish.json"

OUTPUT_DIR = "output/finetuning_shift"
INTERMEDIATE_DIR = "output/finetuning_shift/intermediate"


# ── Utilities ──────────────────────────────────────────────────────────

def a_proj_b(a, b):
    """Scalar projection of a onto b."""
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


def load_eval_questions():
    with open(EVAL_QUESTIONS_PATH, "r") as f:
        data = json.load(f)
    return data["questions"]


def build_prompts(questions, tokenizer):
    """Build neutral and inoculation-present prompts for each question."""
    prompts = []
    for q in questions:
        # Neutral condition
        neutral_msgs = [{"role": "user", "content": q}]
        neutral_text = tokenizer.apply_chat_template(
            neutral_msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(("neutral", q, neutral_text))

        # Inoculation-present condition
        inoc_msgs = [
            {"role": "system", "content": INOCULATION_PROMPT},
            {"role": "user", "content": q},
        ]
        inoc_text = tokenizer.apply_chat_template(
            inoc_msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(("inoculation", q, inoc_text))

    return prompts


def extract_activations(model, tokenizer, prompts, layers):
    """
    For each prompt, extract the last-prompt-token hidden state at each layer.
    Returns dict: {layer: tensor of shape [n_prompts, hidden_dim]}
    """
    results = {l: [] for l in layers}

    for condition, question, text in tqdm(prompts, desc="  Extracting activations"):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for layer in layers:
            # Last token of the prompt (which is the last token since we used add_generation_prompt=True)
            last_hidden = outputs.hidden_states[layer][:, -1, :].detach().cpu().float()
            results[layer].append(last_hidden)
        del outputs
        torch.cuda.empty_cache()

    for layer in layers:
        results[layer] = torch.cat(results[layer], dim=0)  # [n_prompts, hidden_dim]

    return results


def compute_projections(activations, spanish_vectors, caps_vectors, layers):
    """Compute scalar projections onto both trait vectors at each layer."""
    rows = []
    n = activations[layers[0]].shape[0]

    for i in range(n):
        for layer in layers:
            act = activations[layer][i]  # [hidden_dim]
            sp_vec = spanish_vectors[layer].float()
            cp_vec = caps_vectors[layer].float()
            sp_proj = a_proj_b(act, sp_vec).item()
            cp_proj = a_proj_b(act, cp_vec).item()
            rows.append({
                "prompt_idx": i,
                "layer": layer,
                "spanish_proj": sp_proj,
                "caps_proj": cp_proj,
            })
    return rows


# ── Main ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    # Load trait vectors
    print("Loading trait vectors...")
    spanish_vectors = torch.load(SPANISH_VECTOR_PATH, weights_only=False)
    caps_vectors = torch.load(CAPS_VECTOR_PATH, weights_only=False)
    print(f"  Spanish vector shape: {spanish_vectors.shape}")
    print(f"  CAPS vector shape: {caps_vectors.shape}")

    # Load eval questions
    questions = load_eval_questions()
    print(f"Loaded {len(questions)} eval questions")
    print(f"Inoculation prompt: \"{INOCULATION_PROMPT}\"")

    # We'll collect all projection data here
    all_projections = []

    for model_name, model_id in MODELS:
        intermediate_path = os.path.join(INTERMEDIATE_DIR, f"{model_name}_projections.pt")

        # Check for cached intermediate results
        if os.path.exists(intermediate_path):
            print(f"\n{'='*60}")
            print(f"Loading cached results for {model_name}")
            cached = torch.load(intermediate_path, weights_only=False)
            all_projections.extend(cached)
            print(f"  Loaded {len(cached)} projection rows")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {model_name} ({model_id})")
        print(f"{'='*60}")

        try:
            # Load model
            print(f"  Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Build prompts (need tokenizer for chat template)
            prompts = build_prompts(questions, tokenizer)
            print(f"  Built {len(prompts)} prompts (20 questions × 2 conditions)")

            # Extract activations
            activations = extract_activations(model, tokenizer, prompts, LAYERS)

            # Compute projections
            proj_rows = compute_projections(activations, spanish_vectors, caps_vectors, LAYERS)

            # Tag with model name and condition
            conditions = [p[0] for p in prompts]
            question_texts = [p[1] for p in prompts]
            for row in proj_rows:
                pidx = row["prompt_idx"]
                row["model"] = model_name
                row["condition"] = conditions[pidx]
                row["question"] = question_texts[pidx]

            # Save intermediate
            torch.save(proj_rows, intermediate_path)
            print(f"  Saved {len(proj_rows)} projection rows to {intermediate_path}")

            all_projections.extend(proj_rows)

        except Exception as e:
            print(f"  ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        finally:
            # Cleanup GPU memory
            if 'model' in dir():
                try:
                    del model
                except:
                    pass
            if 'tokenizer' in dir():
                try:
                    del tokenizer
                except:
                    pass
            if 'activations' in dir():
                try:
                    del activations
                except:
                    pass
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  GPU memory freed")

    # ── Compute shifts ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Computing finetuning shifts...")
    print(f"{'='*60}")

    df = pd.DataFrame(all_projections)

    # Get base model means per condition per layer
    base_means = df[df["model"] == "Base"].groupby(["condition", "layer"]).agg(
        spanish_proj_mean=("spanish_proj", "mean"),
        caps_proj_mean=("caps_proj", "mean"),
    ).reset_index()

    # Compute shifts for each model
    shift_rows = []
    for model_name in df["model"].unique():
        model_means = df[df["model"] == model_name].groupby(["condition", "layer"]).agg(
            spanish_proj_mean=("spanish_proj", "mean"),
            caps_proj_mean=("caps_proj", "mean"),
        ).reset_index()

        for _, row in model_means.iterrows():
            condition = row["condition"]
            layer = row["layer"]
            base_row = base_means[
                (base_means["condition"] == condition) & (base_means["layer"] == layer)
            ].iloc[0]

            shift_rows.append({
                "Model": model_name,
                "Condition": condition,
                "Layer": layer,
                "Spanish_Shift": row["spanish_proj_mean"] - base_row["spanish_proj_mean"],
                "CAPS_Shift": row["caps_proj_mean"] - base_row["caps_proj_mean"],
                "Spanish_Proj_Mean": row["spanish_proj_mean"],
                "CAPS_Proj_Mean": row["caps_proj_mean"],
            })

    shift_df = pd.DataFrame(shift_rows)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "finetuning_shifts.csv")
    shift_df.to_csv(csv_path, index=False)
    print(f"\nShift table saved to {csv_path}")

    # Also save full projections CSV
    full_csv_path = os.path.join(OUTPUT_DIR, "all_projections.csv")
    df.to_csv(full_csv_path, index=False)
    print(f"Full projections saved to {full_csv_path}")

    # Print the summary table
    print(f"\n{'='*60}")
    print("FINETUNING SHIFT SUMMARY")
    print(f"{'='*60}")

    # Pivot for display — focus on key layers (24 and 32)
    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")
        layer_df = shift_df[shift_df["Layer"] == layer].pivot(
            index="Model", columns="Condition",
            values=["Spanish_Shift", "CAPS_Shift"]
        )
        # Flatten column names
        layer_df.columns = [f"{val}_{cond}" for val, cond in layer_df.columns]
        print(layer_df.to_string())

    # Print compact table for key layers
    print(f"\n\n{'='*60}")
    print("COMPACT SUMMARY (Layers 24 & 32)")
    print(f"{'='*60}")
    key_df = shift_df[shift_df["Layer"].isin([24, 32])].sort_values(["Layer", "Model", "Condition"])
    print(key_df.to_string(index=False))


if __name__ == "__main__":
    main()
