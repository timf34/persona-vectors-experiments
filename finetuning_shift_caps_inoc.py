"""
Finetuning Shift Analysis — CAPS Inoculation Prompt Rerun

Reruns Step 3 for Base, AllCapsInoc, and SpanishInoc using the CAPS
inoculation prompt instead of the Spanish one. This tests whether
context-gating is specific to the matched inoculation prompt.

Conditions:
  - neutral: same as before (no system prompt)
  - caps_inoculation: system prompt = CAPS instruction #1 from caps.json
"""

import os
import json
import gc

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────

MODELS = [
    ("Base",        "Qwen/Qwen2.5-32B-Instruct"),
    ("AllCapsInoc", "longtermrisk/Qwen2.5-32B-Instruct-ftjob-271c92c27ec5"),
    ("SpanishInoc", "longtermrisk/Qwen2.5-32B-Instruct-ftjob-50abc9e9a009"),
]

LAYERS = [16, 24, 32, 48]

CAPS_INOCULATION_PROMPT = "You ALWAYS write your entire response in ALL CAPITAL LETTERS. Every single letter must be uppercase."

SPANISH_VECTOR_PATH = "persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt"
CAPS_VECTOR_PATH    = "persona_vectors/Qwen2.5-32B-Instruct/caps_response_avg_diff.pt"

EVAL_QUESTIONS_PATH = "data_generation/trait_data_eval/spanish.json"

OUTPUT_DIR = "output/finetuning_shift_caps_inoc"
INTERMEDIATE_DIR = "output/finetuning_shift_caps_inoc/intermediate"


# ── Utilities ──────────────────────────────────────────────────────────

def a_proj_b(a, b):
    """Scalar projection of a onto b."""
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


def load_eval_questions():
    with open(EVAL_QUESTIONS_PATH, "r") as f:
        data = json.load(f)
    return data["questions"]


def build_prompts(questions, tokenizer):
    """Build neutral and CAPS-inoculation-present prompts for each question."""
    prompts = []
    for q in questions:
        # Neutral condition
        neutral_msgs = [{"role": "user", "content": q}]
        neutral_text = tokenizer.apply_chat_template(
            neutral_msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(("neutral", q, neutral_text))

        # CAPS inoculation-present condition
        inoc_msgs = [
            {"role": "system", "content": CAPS_INOCULATION_PROMPT},
            {"role": "user", "content": q},
        ]
        inoc_text = tokenizer.apply_chat_template(
            inoc_msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(("caps_inoculation", q, inoc_text))

    return prompts


def extract_activations(model, tokenizer, prompts, layers):
    results = {l: [] for l in layers}

    for condition, question, text in tqdm(prompts, desc="  Extracting activations"):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for layer in layers:
            last_hidden = outputs.hidden_states[layer][:, -1, :].detach().cpu().float()
            results[layer].append(last_hidden)
        del outputs
        torch.cuda.empty_cache()

    for layer in layers:
        results[layer] = torch.cat(results[layer], dim=0)

    return results


def compute_projections(activations, spanish_vectors, caps_vectors, layers):
    rows = []
    n = activations[layers[0]].shape[0]

    for i in range(n):
        for layer in layers:
            act = activations[layer][i]
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
    print(f"CAPS inoculation prompt: \"{CAPS_INOCULATION_PROMPT}\"")

    all_projections = []

    for model_name, model_id in MODELS:
        intermediate_path = os.path.join(INTERMEDIATE_DIR, f"{model_name}_projections.pt")

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
            print(f"  Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
            # Load tokenizer from local cache to avoid HF API rate limits
            tokenizer = AutoTokenizer.from_pretrained(
                "/workspace/hf_cache/hub/models--longtermrisk--Qwen2.5-32B-Instruct-ftjob-271c92c27ec5/snapshots/40f07110d1912a38b19e052de6a57f0f7e505c91/"
            )

            prompts = build_prompts(questions, tokenizer)
            print(f"  Built {len(prompts)} prompts (20 questions × 2 conditions)")

            activations = extract_activations(model, tokenizer, prompts, LAYERS)
            proj_rows = compute_projections(activations, spanish_vectors, caps_vectors, LAYERS)

            conditions = [p[0] for p in prompts]
            question_texts = [p[1] for p in prompts]
            for row in proj_rows:
                pidx = row["prompt_idx"]
                row["model"] = model_name
                row["condition"] = conditions[pidx]
                row["question"] = question_texts[pidx]

            torch.save(proj_rows, intermediate_path)
            print(f"  Saved {len(proj_rows)} projection rows to {intermediate_path}")

            all_projections.extend(proj_rows)

        except Exception as e:
            print(f"  ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        finally:
            for var_name in ['model', 'tokenizer', 'activations']:
                if var_name in locals():
                    try:
                        del locals()[var_name]
                    except:
                        pass
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  GPU memory freed")

    # ── Compute shifts ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Computing finetuning shifts (CAPS inoculation prompt)...")
    print(f"{'='*60}")

    df = pd.DataFrame(all_projections)

    base_means = df[df["model"] == "Base"].groupby(["condition", "layer"]).agg(
        spanish_proj_mean=("spanish_proj", "mean"),
        caps_proj_mean=("caps_proj", "mean"),
    ).reset_index()

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

    csv_path = os.path.join(OUTPUT_DIR, "finetuning_shifts_caps_inoc.csv")
    shift_df.to_csv(csv_path, index=False)
    print(f"\nShift table saved to {csv_path}")

    full_csv_path = os.path.join(OUTPUT_DIR, "all_projections_caps_inoc.csv")
    df.to_csv(full_csv_path, index=False)
    print(f"Full projections saved to {full_csv_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("FINETUNING SHIFT SUMMARY (CAPS Inoculation Prompt)")
    print(f"{'='*60}")

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")
        layer_df = shift_df[shift_df["Layer"] == layer].pivot(
            index="Model", columns="Condition",
            values=["Spanish_Shift", "CAPS_Shift"]
        )
        layer_df.columns = [f"{val}_{cond}" for val, cond in layer_df.columns]
        print(layer_df.to_string())

    # ── Statistical comparison ─────────────────────────────────────
    import numpy as np
    from scipy import stats

    print(f"\n\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")

    # Build per-prompt shifts
    base_lookup = {}
    for _, r in df[(df["model"] == "Base") & (df["layer"].isin([24, 32]))].iterrows():
        base_lookup[(r["condition"], r["layer"], r["prompt_idx"])] = (r["spanish_proj"], r["caps_proj"])

    prompt_shifts = []
    for _, r in df[(df["model"] != "Base") & (df["layer"].isin([24, 32]))].iterrows():
        key = (r["condition"], r["layer"], r["prompt_idx"])
        if key in base_lookup:
            b_sp, b_cp = base_lookup[key]
            prompt_shifts.append({
                "model": r["model"],
                "condition": r["condition"],
                "layer": r["layer"],
                "prompt_idx": r["prompt_idx"],
                "spanish_shift": r["spanish_proj"] - b_sp,
                "caps_shift": r["caps_proj"] - b_cp,
            })

    ps = pd.DataFrame(prompt_shifts)

    # Mean ± SE
    print(f"\nMean ± SE (N=20 per cell):")
    for layer in [24, 32]:
        print(f"\n  Layer {layer}:")
        print(f"  {'Model':<15} {'Condition':<18} {'Spanish Shift':>18} {'CAPS Shift':>18}")
        print(f"  {'-'*72}")
        for model in ["AllCapsInoc", "SpanishInoc"]:
            for cond in ["neutral", "caps_inoculation"]:
                sub = ps[(ps["model"] == model) & (ps["condition"] == cond) & (ps["layer"] == layer)]
                n = len(sub)
                sp_m = sub["spanish_shift"].mean()
                sp_se = sub["spanish_shift"].std(ddof=1) / np.sqrt(n)
                cp_m = sub["caps_shift"].mean()
                cp_se = sub["caps_shift"].std(ddof=1) / np.sqrt(n)
                print(f"  {model:<15} {cond:<18} {sp_m:>7.2f} ± {sp_se:<6.2f}  {cp_m:>7.2f} ± {cp_se:<6.2f}")

    # Paired t-tests: within-model neutral vs caps_inoculation
    print(f"\n\nPaired t-tests: neutral vs caps_inoculation (within model)")
    for layer in [24, 32]:
        print(f"\n  Layer {layer}:")
        for model in ["AllCapsInoc", "SpanishInoc"]:
            for trait in ["spanish_shift", "caps_shift"]:
                neut = ps[(ps["model"] == model) & (ps["condition"] == "neutral") & (ps["layer"] == layer)].sort_values("prompt_idx")[trait].values
                inoc = ps[(ps["model"] == model) & (ps["condition"] == "caps_inoculation") & (ps["layer"] == layer)].sort_values("prompt_idx")[trait].values
                t_stat, p_val = stats.ttest_rel(neut, inoc)
                diff = inoc.mean() - neut.mean()
                diff_std = (inoc - neut).std(ddof=1)
                d = diff / diff_std if diff_std > 0 else 0
                label = "Spanish" if "spanish" in trait else "CAPS"
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  {model:<15} {label:<8}  inoc-neut={diff:>+7.2f}  t={t_stat:>+7.2f}  p={p_val:.4f} {sig}  d={d:>+5.2f}")


if __name__ == "__main__":
    main()
