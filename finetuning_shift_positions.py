"""
Finetuning Shift Analysis — Token Position Comparison

Extracts activations at three token positions per prompt:
  1. Last prompt token (from prefill step)
  2. First response token (generation step 1)
  3. Mean of first 5 response tokens (generation steps 1-5)

Projects onto Spanish and CAPS trait vectors, computes shifts relative
to the base model. Tests whether CAPS suppression in AllCapsInoc appears
only at the prompt token or also at response-token positions.
"""

import os
import json
import gc

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────

MODELS = [
    ("Base",                 "Qwen/Qwen2.5-32B-Instruct"),
    ("Baseline_NoInoc",      "longtermrisk/Qwen2.5-32B-Instruct-ftjob-abd8475aaeed"),
    ("SpanishInoc",          "longtermrisk/Qwen2.5-32B-Instruct-ftjob-50abc9e9a009"),
    ("AllCapsInoc",          "longtermrisk/Qwen2.5-32B-Instruct-ftjob-271c92c27ec5"),
    ("RephrasedSpanishInoc", "longtermrisk/Qwen2.5-32B-Instruct-ftjob-b2d69a1ba642"),
    ("IrrelevantHoney",      "longtermrisk/Qwen2.5-32B-Instruct-ftjob-466a1ba110e4"),
    ("Control",              "longtermrisk/Qwen2.5-32B-Instruct-ftjob-b0fafb674e38"),
]

LAYERS = [16, 24, 32]

SPANISH_INOC_PROMPT = "You always speak in Spanish. Respond only in Spanish, do not use any English words."
CAPS_INOC_PROMPT = "You ALWAYS write your entire response in ALL CAPITAL LETTERS. Every single letter must be uppercase."

# Models that get the extra CAPS inoculation condition
CAPS_INOC_MODELS = {"AllCapsInoc", "SpanishInoc"}

SPANISH_VECTOR_PATH = "persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt"
CAPS_VECTOR_PATH    = "persona_vectors/Qwen2.5-32B-Instruct/caps_response_avg_diff.pt"

EVAL_QUESTIONS_PATH = "data_generation/trait_data_eval/spanish.json"

# Use local tokenizer to avoid HF rate limits
TOKENIZER_PATH = "/workspace/hf_cache/hub/models--longtermrisk--Qwen2.5-32B-Instruct-ftjob-271c92c27ec5/snapshots/40f07110d1912a38b19e052de6a57f0f7e505c91/"

OUTPUT_DIR = "output/finetuning_shift_positions"
INTERMEDIATE_DIR = "output/finetuning_shift_positions/intermediate"


# ── Utilities ──────────────────────────────────────────────────────────

def a_proj_b(a, b):
    """Scalar projection of a onto b."""
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


def load_eval_questions():
    with open(EVAL_QUESTIONS_PATH, "r") as f:
        data = json.load(f)
    return data["questions"]


def build_prompts(questions, tokenizer):
    """Build prompts for all conditions."""
    prompts = []
    for q_idx, q in enumerate(questions):
        # Neutral
        neutral_msgs = [{"role": "user", "content": q}]
        neutral_text = tokenizer.apply_chat_template(
            neutral_msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(("neutral", q_idx, q, neutral_text))

        # Spanish inoculation
        sp_msgs = [
            {"role": "system", "content": SPANISH_INOC_PROMPT},
            {"role": "user", "content": q},
        ]
        sp_text = tokenizer.apply_chat_template(
            sp_msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(("spanish_inoculation", q_idx, q, sp_text))

    return prompts


def build_caps_inoc_prompts(questions, tokenizer):
    """Build CAPS inoculation condition prompts (for AllCapsInoc/SpanishInoc only)."""
    prompts = []
    for q_idx, q in enumerate(questions):
        caps_msgs = [
            {"role": "system", "content": CAPS_INOC_PROMPT},
            {"role": "user", "content": q},
        ]
        caps_text = tokenizer.apply_chat_template(
            caps_msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(("caps_inoculation", q_idx, q, caps_text))
    return prompts


def extract_positions(model, tokenizer, prompts, layers, spanish_vectors, caps_vectors):
    """
    For each prompt, run generate(max_new_tokens=5) and extract activations
    at three positions: last_prompt, first_response, mean_5_response.
    Returns list of projection row dicts.
    """
    rows = []

    for condition, q_idx, question, text in tqdm(prompts, desc="  Generating"):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=6,
                min_new_tokens=6,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        # outputs.hidden_states is a tuple of steps:
        # Step 0: prefill — hidden_states[0][layer] has shape [1, prompt_len, hidden_dim]
        # Step i (i>=1): generation step — hidden_states[i][layer] has shape [1, 1, hidden_dim]
        n_gen_steps = len(outputs.hidden_states) - 1  # exclude prefill

        for layer in layers:
            sp_vec = spanish_vectors[layer].float()
            cp_vec = caps_vectors[layer].float()

            # 1. Last prompt token
            last_prompt = outputs.hidden_states[0][layer][:, -1, :].detach().cpu().float().squeeze(0)
            sp_proj_lp = a_proj_b(last_prompt, sp_vec).item()
            cp_proj_lp = a_proj_b(last_prompt, cp_vec).item()
            rows.append({
                "condition": condition, "question_idx": q_idx, "layer": layer,
                "position": "last_prompt",
                "spanish_proj": sp_proj_lp, "caps_proj": cp_proj_lp,
            })

            # 2. First response token (generation step 1)
            first_resp = outputs.hidden_states[1][layer][:, 0, :].detach().cpu().float().squeeze(0)
            sp_proj_fr = a_proj_b(first_resp, sp_vec).item()
            cp_proj_fr = a_proj_b(first_resp, cp_vec).item()
            rows.append({
                "condition": condition, "question_idx": q_idx, "layer": layer,
                "position": "first_response",
                "spanish_proj": sp_proj_fr, "caps_proj": cp_proj_fr,
            })

            # 3. Mean of first 5 response tokens (use however many are available, up to 5)
            n_resp = min(5, n_gen_steps)
            resp_hiddens = []
            for step in range(1, n_resp + 1):
                h = outputs.hidden_states[step][layer][:, 0, :].detach().cpu().float().squeeze(0)
                resp_hiddens.append(h)
            mean_5 = torch.stack(resp_hiddens).mean(dim=0)
            sp_proj_m5 = a_proj_b(mean_5, sp_vec).item()
            cp_proj_m5 = a_proj_b(mean_5, cp_vec).item()
            rows.append({
                "condition": condition, "question_idx": q_idx, "layer": layer,
                "position": "mean_5_response",
                "spanish_proj": sp_proj_m5, "caps_proj": cp_proj_m5,
            })

        del outputs
        torch.cuda.empty_cache()

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

    # Load eval questions and tokenizer
    questions = load_eval_questions()
    print(f"Loaded {len(questions)} eval questions")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Build prompts (same tokenizer for all models)
    standard_prompts = build_prompts(questions, tokenizer)
    caps_inoc_prompts = build_caps_inoc_prompts(questions, tokenizer)
    print(f"  Standard prompts: {len(standard_prompts)} (20 questions × 2 conditions)")
    print(f"  CAPS inoc prompts: {len(caps_inoc_prompts)} (20 questions × 1 condition)")

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

            # Standard conditions (neutral + spanish inoc)
            prompts_to_run = standard_prompts.copy()

            # Add CAPS inoc condition for selected models
            if model_name in CAPS_INOC_MODELS:
                prompts_to_run.extend(caps_inoc_prompts)
                print(f"  Including CAPS inoculation condition ({len(prompts_to_run)} total prompts)")
            else:
                print(f"  {len(prompts_to_run)} prompts")

            proj_rows = extract_positions(
                model, tokenizer, prompts_to_run, LAYERS,
                spanish_vectors, caps_vectors
            )

            # Tag with model name
            for row in proj_rows:
                row["model"] = model_name

            torch.save(proj_rows, intermediate_path)
            print(f"  Saved {len(proj_rows)} projection rows to {intermediate_path}")

            all_projections.extend(proj_rows)

        except Exception as e:
            print(f"  ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        finally:
            if 'model' in dir():
                try:
                    del model
                except:
                    pass
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  GPU memory freed")

    # ── Compute shifts and statistics ──────────────────────────────
    print(f"\n{'='*60}")
    print("Computing shifts and statistics...")
    print(f"{'='*60}")

    df = pd.DataFrame(all_projections)

    # Save full projections
    full_csv = os.path.join(OUTPUT_DIR, "all_projections_positions.csv")
    df.to_csv(full_csv, index=False)
    print(f"Full projections saved to {full_csv}")

    # Build base lookup: (condition, layer, position, question_idx) -> (sp, cp)
    base = df[df["model"] == "Base"]
    base_lookup = {}
    for _, r in base.iterrows():
        key = (r["condition"], r["layer"], r["position"], r["question_idx"])
        base_lookup[key] = (r["spanish_proj"], r["caps_proj"])

    # Compute per-prompt shifts
    shift_rows = []
    for _, r in df[df["model"] != "Base"].iterrows():
        key = (r["condition"], r["layer"], r["position"], r["question_idx"])
        if key in base_lookup:
            b_sp, b_cp = base_lookup[key]
            shift_rows.append({
                "model": r["model"],
                "condition": r["condition"],
                "layer": r["layer"],
                "position": r["position"],
                "question_idx": r["question_idx"],
                "spanish_shift": r["spanish_proj"] - b_sp,
                "caps_shift": r["caps_proj"] - b_cp,
            })

    shifts = pd.DataFrame(shift_rows)

    # Save shifts
    shifts_csv = os.path.join(OUTPUT_DIR, "per_prompt_shifts.csv")
    shifts.to_csv(shifts_csv, index=False)

    # Summary table: mean ± SE
    from scipy import stats

    print(f"\n{'='*90}")
    print("FINETUNING SHIFTS BY TOKEN POSITION: Mean ± SE (N=20)")
    print(f"{'='*90}")

    positions = ["last_prompt", "first_response", "mean_5_response"]
    all_models = df["model"].unique()
    finetuned_models = [m for m in all_models if m != "Base"]

    for layer in LAYERS:
        print(f"\n{'─'*90}")
        print(f"LAYER {layer}")
        print(f"{'─'*90}")
        print(f"{'Model':<22} {'Condition':<20} {'Position':<18} {'Spanish Shift':>16} {'CAPS Shift':>16}")
        print(f"{'─'*90}")

        for model in finetuned_models:
            for cond in sorted(shifts[shifts["model"] == model]["condition"].unique()):
                for pos in positions:
                    sub = shifts[
                        (shifts["model"] == model) &
                        (shifts["condition"] == cond) &
                        (shifts["layer"] == layer) &
                        (shifts["position"] == pos)
                    ]
                    if len(sub) == 0:
                        continue
                    n = len(sub)
                    sp_m = sub["spanish_shift"].mean()
                    sp_se = sub["spanish_shift"].std(ddof=1) / np.sqrt(n)
                    cp_m = sub["caps_shift"].mean()
                    cp_se = sub["caps_shift"].std(ddof=1) / np.sqrt(n)
                    print(f"{model:<22} {cond:<20} {pos:<18} {sp_m:>6.2f} ± {sp_se:<5.2f}  {cp_m:>6.2f} ± {cp_se:<5.2f}")
                print()

    # ── Key comparison: AllCapsInoc CAPS shift by position ──────────
    print(f"\n{'='*90}")
    print("KEY COMPARISON: AllCapsInoc CAPS shift by position")
    print(f"{'='*90}")
    print("(Does CAPS shift appear at response tokens even though suppressed at prompt token?)\n")

    for layer in LAYERS:
        print(f"Layer {layer}:")
        for cond in ["neutral", "spanish_inoculation", "caps_inoculation"]:
            sub_model = shifts[
                (shifts["model"] == "AllCapsInoc") &
                (shifts["condition"] == cond) &
                (shifts["layer"] == layer)
            ]
            if len(sub_model) == 0:
                continue
            print(f"  {cond}:")
            for pos in positions:
                sub = sub_model[sub_model["position"] == pos]
                if len(sub) == 0:
                    continue
                n = len(sub)
                cp_m = sub["caps_shift"].mean()
                cp_se = sub["caps_shift"].std(ddof=1) / np.sqrt(n)
                print(f"    {pos:<20} CAPS shift = {cp_m:>7.2f} ± {cp_se:.2f}")
        print()

    # ── Paired t-tests: last_prompt vs first_response for CAPS ─────
    print(f"\n{'='*90}")
    print("PAIRED t-tests: last_prompt vs first_response (CAPS shift)")
    print(f"{'='*90}")
    print("(Tests whether CAPS shift changes from prompt to response token)\n")

    for model in ["AllCapsInoc", "SpanishInoc", "Baseline_NoInoc"]:
        print(f"{model}:")
        for layer in LAYERS:
            for cond in sorted(shifts[shifts["model"] == model]["condition"].unique()):
                lp = shifts[
                    (shifts["model"] == model) & (shifts["condition"] == cond) &
                    (shifts["layer"] == layer) & (shifts["position"] == "last_prompt")
                ].sort_values("question_idx")["caps_shift"].values

                fr = shifts[
                    (shifts["model"] == model) & (shifts["condition"] == cond) &
                    (shifts["layer"] == layer) & (shifts["position"] == "first_response")
                ].sort_values("question_idx")["caps_shift"].values

                if len(lp) == 0 or len(fr) == 0:
                    continue

                t_stat, p_val = stats.ttest_rel(lp, fr)
                diff = fr.mean() - lp.mean()
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  L{layer} {cond:<20} lp={lp.mean():>6.2f}  fr={fr.mean():>6.2f}  diff={diff:>+6.2f}  p={p_val:.4f} {sig}")
        print()


if __name__ == "__main__":
    main()
