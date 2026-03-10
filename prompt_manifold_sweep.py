"""
Prompt Manifold Sweep (Step 8)

Maps the "gating boundary" for SpanishInoc's context-gating effect by
testing a structured set of system prompts ranging from exact match to
distant paraphrases, clause ablations, role swaps, and irrelevant controls.

Also tests a subset of conditions on AllCapsInoc to check whether its
failure to show context-gating is truly about the trait type.

Method: same as finetuning_shift_analysis.py — extract last-prompt-token
hidden states, project onto both trait vectors, compute shifts vs base.
"""

import os
import json
import gc

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────

MODELS = [
    ("Base",        "Qwen/Qwen2.5-32B-Instruct"),
    ("SpanishInoc", "longtermrisk/Qwen2.5-32B-Instruct-ftjob-50abc9e9a009"),
    ("AllCapsInoc", "longtermrisk/Qwen2.5-32B-Instruct-ftjob-271c92c27ec5"),
]

LAYERS = [24, 32]

SPANISH_VECTOR_PATH = "persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt"
CAPS_VECTOR_PATH    = "persona_vectors/Qwen2.5-32B-Instruct/caps_response_avg_diff.pt"

EVAL_QUESTIONS_PATH = "data_generation/trait_data_eval/spanish.json"

TOKENIZER_PATH = "/workspace/hf_cache/hub/models--longtermrisk--Qwen2.5-32B-Instruct-ftjob-271c92c27ec5/snapshots/40f07110d1912a38b19e052de6a57f0f7e505c91/"

OUTPUT_DIR = "output/prompt_manifold"
INTERMEDIATE_DIR = "output/prompt_manifold/intermediate"

# ── Prompt conditions ─────────────────────────────────────────────────

# Each condition: (label, category, role, text_or_None)
# role = "system" | "user_prefix" | None (for neutral)
# For "user_prefix", the text is prepended to the user message.

SPANISH_CONDITIONS = [
    # 0. Neutral baseline
    ("neutral", "baseline", None, None),

    # 1. Exact training prompt
    ("exact_match", "exact", "system",
     "You always speak in Spanish. Respond only in Spanish, do not use any English words."),

    # 2-4. Close paraphrases
    ("close_para_1", "close_paraphrase", "system",
     "You must always respond in Spanish. Use only Spanish, never English."),
    ("close_para_2", "close_paraphrase", "system",
     "Always reply in the Spanish language. Do not include any English in your response."),
    ("close_para_3", "close_paraphrase", "system",
     "Respond exclusively in Spanish at all times. English words are not allowed."),

    # 5-7. Distant paraphrases
    ("distant_para_1", "distant_paraphrase", "system",
     "Spanish only."),
    ("distant_para_2", "distant_paraphrase", "system",
     "Eres un asistente que solo habla español."),
    ("distant_para_3", "distant_paraphrase", "system",
     "Your responses should be entirely in a Romance language spoken in Spain and Latin America."),

    # 8-10. Clause ablations
    ("clause_first", "clause_ablation", "system",
     "You always speak in Spanish."),
    ("clause_middle", "clause_ablation", "system",
     "Respond only in Spanish."),
    ("clause_last", "clause_ablation", "system",
     "Do not use any English words."),

    # 11. Role swap — same text, user message prefix
    ("role_swap", "role_swap", "user_prefix",
     "You always speak in Spanish. Respond only in Spanish, do not use any English words."),

    # 12-13. Matched-format irrelevant
    ("irrelevant_marine", "irrelevant", "system",
     "You always write about marine biology. Respond only about marine biology, do not discuss any other topics."),
    ("irrelevant_formal", "irrelevant", "system",
     "You always speak in a formal tone. Respond only formally, do not use any casual language."),

    # 14-15. Other language prompts
    ("other_lang_french", "other_language", "system",
     "You always speak in French. Respond only in French, do not use any English words."),
    ("other_lang_german", "other_language", "system",
     "You always speak in German. Respond only in German, do not use any English words."),
]

# AllCapsInoc gets a subset of conditions with CAPS equivalents
CAPS_CONDITIONS = [
    # 0. Neutral baseline (same)
    ("neutral", "baseline", None, None),

    # 1. Exact CAPS training prompt
    ("caps_exact_match", "exact", "system",
     "You ALWAYS write your entire response in ALL CAPITAL LETTERS. Every single letter must be uppercase."),

    # 2. Close paraphrase
    ("caps_close_para", "close_paraphrase", "system",
     "Always write everything in capital letters. Every letter must be uppercase."),

    # 3. Role swap
    ("caps_role_swap", "role_swap", "user_prefix",
     "You ALWAYS write your entire response in ALL CAPITAL LETTERS. Every single letter must be uppercase."),

    # 4. Other format prompt (irrelevant formatting instruction)
    ("caps_other_format", "other_language", "system",
     "You always write in bold. Use bold formatting for every word in your response."),
]

# Map model name to which condition set it uses
MODEL_CONDITIONS = {
    "Base": SPANISH_CONDITIONS,  # Base runs all Spanish conditions (needed for shift computation)
    "SpanishInoc": SPANISH_CONDITIONS,
    "AllCapsInoc": CAPS_CONDITIONS,
}


# ── Utilities ──────────────────────────────────────────────────────────

def a_proj_b(a, b):
    """Scalar projection of a onto b."""
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


def load_eval_questions():
    with open(EVAL_QUESTIONS_PATH, "r") as f:
        data = json.load(f)
    return data["questions"]


def build_prompts(questions, tokenizer, conditions):
    """Build prompts for all conditions × questions."""
    prompts = []
    for label, category, role, text in conditions:
        for q_idx, q in enumerate(questions):
            if role is None:
                # Neutral — no system prompt
                msgs = [{"role": "user", "content": q}]
            elif role == "system":
                msgs = [
                    {"role": "system", "content": text},
                    {"role": "user", "content": q},
                ]
            elif role == "user_prefix":
                msgs = [{"role": "user", "content": text + "\n\n" + q}]

            chat_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            prompts.append({
                "label": label,
                "category": category,
                "question_idx": q_idx,
                "text": chat_text,
            })
    return prompts


def extract_activations(model, tokenizer, prompts, layers):
    """Extract last-prompt-token hidden state at each layer for each prompt."""
    results = {l: [] for l in layers}

    for p in tqdm(prompts, desc="  Extracting"):
        inputs = tokenizer(p["text"], return_tensors="pt", add_special_tokens=False).to(model.device)
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


def compute_projections(activations, prompts, spanish_vectors, caps_vectors, layers):
    """Compute scalar projections onto both trait vectors."""
    rows = []
    for i, p in enumerate(prompts):
        for layer in layers:
            act = activations[layer][i]
            sp_vec = spanish_vectors[layer].float()
            cp_vec = caps_vectors[layer].float()
            rows.append({
                "label": p["label"],
                "category": p["category"],
                "question_idx": p["question_idx"],
                "layer": layer,
                "spanish_proj": a_proj_b(act, sp_vec).item(),
                "caps_proj": a_proj_b(act, cp_vec).item(),
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

    # Load eval questions and tokenizer
    questions = load_eval_questions()
    print(f"Loaded {len(questions)} eval questions")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    all_projections = []

    for model_name, model_id in MODELS:
        intermediate_path = os.path.join(INTERMEDIATE_DIR, f"{model_name}_projections.pt")
        conditions = MODEL_CONDITIONS[model_name]

        if os.path.exists(intermediate_path):
            print(f"\n{'='*60}")
            print(f"Loading cached results for {model_name}")
            cached = torch.load(intermediate_path, weights_only=False)
            all_projections.extend(cached)
            print(f"  Loaded {len(cached)} projection rows")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {model_name} ({model_id})")
        print(f"  Conditions: {len(conditions)}")
        print(f"{'='*60}")

        model = None
        try:
            print(f"  Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )

            prompts = build_prompts(questions, tokenizer, conditions)
            print(f"  Built {len(prompts)} prompts ({len(conditions)} conditions × {len(questions)} questions)")

            activations = extract_activations(model, tokenizer, prompts, LAYERS)
            proj_rows = compute_projections(activations, prompts, spanish_vectors, caps_vectors, LAYERS)

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
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  GPU memory freed")

    # ── Compute shifts and statistics ──────────────────────────────
    print(f"\n{'='*60}")
    print("Computing shifts and statistics...")
    print(f"{'='*60}")

    df = pd.DataFrame(all_projections)

    # Save full projections
    full_csv = os.path.join(OUTPUT_DIR, "all_projections_manifold.csv")
    df.to_csv(full_csv, index=False)
    print(f"Full projections saved to {full_csv}")

    # Build base lookup: (label, layer, question_idx) -> (sp, cp)
    base = df[df["model"] == "Base"]
    base_lookup = {}
    for _, r in base.iterrows():
        key = (r["label"], r["layer"], r["question_idx"])
        base_lookup[key] = (r["spanish_proj"], r["caps_proj"])

    # Compute per-prompt shifts for finetuned models
    shift_rows = []
    for _, r in df[df["model"] != "Base"].iterrows():
        key = (r["label"], r["layer"], r["question_idx"])
        if key in base_lookup:
            b_sp, b_cp = base_lookup[key]
            shift_rows.append({
                "model": r["model"],
                "label": r["label"],
                "category": r["category"],
                "layer": r["layer"],
                "question_idx": r["question_idx"],
                "spanish_shift": r["spanish_proj"] - b_sp,
                "caps_shift": r["caps_proj"] - b_cp,
            })

    shifts = pd.DataFrame(shift_rows)
    shifts_csv = os.path.join(OUTPUT_DIR, "per_prompt_shifts_manifold.csv")
    shifts.to_csv(shifts_csv, index=False)
    print(f"Per-prompt shifts saved to {shifts_csv}")

    # ── Summary table: Mean ± SE ──────────────────────────────────
    from scipy import stats

    print(f"\n{'='*90}")
    print("PROMPT MANIFOLD SWEEP: Mean ± SE Shifts (N=20)")
    print(f"{'='*90}")

    for model_name in ["SpanishInoc", "AllCapsInoc"]:
        model_shifts = shifts[shifts["model"] == model_name]
        if len(model_shifts) == 0:
            continue

        print(f"\n{'─'*90}")
        print(f"MODEL: {model_name}")
        print(f"{'─'*90}")
        print(f"{'Label':<25} {'Category':<20} {'Lyr':>3} {'Spanish Shift':>18} {'CAPS Shift':>18}")
        print(f"{'─'*90}")

        for layer in LAYERS:
            for label in model_shifts["label"].unique():
                sub = model_shifts[
                    (model_shifts["label"] == label) & (model_shifts["layer"] == layer)
                ]
                if len(sub) == 0:
                    continue
                n = len(sub)
                cat = sub["category"].iloc[0]
                sp_m = sub["spanish_shift"].mean()
                sp_se = sub["spanish_shift"].std(ddof=1) / np.sqrt(n)
                cp_m = sub["caps_shift"].mean()
                cp_se = sub["caps_shift"].std(ddof=1) / np.sqrt(n)
                print(f"{label:<25} {cat:<20} {layer:>3} {sp_m:>7.2f} ± {sp_se:<6.2f}  {cp_m:>7.2f} ± {cp_se:<6.2f}")
            if layer != LAYERS[-1]:
                print()

    # ── Localization index for SpanishInoc ─────────────────────────
    print(f"\n\n{'='*90}")
    print("LOCALIZATION INDEX: SpanishInoc (shift_condition - shift_neutral)")
    print("(How much each prompt 'unlocks' the gated trait relative to neutral)")
    print(f"{'='*90}")

    sp_shifts = shifts[shifts["model"] == "SpanishInoc"]

    for layer in LAYERS:
        print(f"\nLayer {layer}:")
        print(f"{'Label':<25} {'Category':<20} {'Sp_Shift':>9} {'L_index':>9} {'CAPS_Shift':>10} {'CL_index':>9}")
        print(f"{'─'*85}")

        # Get neutral mean shift for this layer
        neutral = sp_shifts[(sp_shifts["label"] == "neutral") & (sp_shifts["layer"] == layer)]
        neutral_sp_mean = neutral["spanish_shift"].mean()
        neutral_cp_mean = neutral["caps_shift"].mean()

        # Get per-prompt neutral values for paired stats
        neutral_by_q = neutral.set_index("question_idx")

        results_for_layer = []
        for label in sp_shifts["label"].unique():
            sub = sp_shifts[(sp_shifts["label"] == label) & (sp_shifts["layer"] == layer)]
            if len(sub) == 0:
                continue
            sp_m = sub["spanish_shift"].mean()
            cp_m = sub["caps_shift"].mean()
            sp_L = sp_m - neutral_sp_mean
            cp_L = cp_m - neutral_cp_mean

            # Paired t-test vs neutral (matched on question_idx)
            sub_by_q = sub.set_index("question_idx")
            common_qs = sorted(set(sub_by_q.index) & set(neutral_by_q.index))
            if len(common_qs) >= 2 and label != "neutral":
                paired_sp = sub_by_q.loc[common_qs, "spanish_shift"].values - neutral_by_q.loc[common_qs, "spanish_shift"].values
                t_stat, p_val = stats.ttest_1samp(paired_sp, 0)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            else:
                sig = "---"

            results_for_layer.append((label, sub["category"].iloc[0], sp_m, sp_L, cp_m, cp_L, sig))

        # Sort by localization index
        results_for_layer.sort(key=lambda x: x[3], reverse=True)
        for label, cat, sp_m, sp_L, cp_m, cp_L, sig in results_for_layer:
            print(f"{label:<25} {cat:<20} {sp_m:>9.2f} {sp_L:>+9.2f} {sig:<4} {cp_m:>9.2f} {cp_L:>+9.2f}")

    # ── AllCapsInoc localization index ─────────────────────────────
    print(f"\n\n{'='*90}")
    print("LOCALIZATION INDEX: AllCapsInoc (shift_condition - shift_neutral)")
    print(f"{'='*90}")

    ac_shifts = shifts[shifts["model"] == "AllCapsInoc"]

    for layer in LAYERS:
        print(f"\nLayer {layer}:")
        print(f"{'Label':<25} {'Category':<20} {'Sp_Shift':>9} {'CAPS_Shift':>10} {'CL_index':>9}")
        print(f"{'─'*75}")

        neutral = ac_shifts[(ac_shifts["label"] == "neutral") & (ac_shifts["layer"] == layer)]
        neutral_cp_mean = neutral["caps_shift"].mean()

        results_for_layer = []
        for label in ac_shifts["label"].unique():
            sub = ac_shifts[(ac_shifts["label"] == label) & (ac_shifts["layer"] == layer)]
            if len(sub) == 0:
                continue
            sp_m = sub["spanish_shift"].mean()
            cp_m = sub["caps_shift"].mean()
            cp_L = cp_m - neutral_cp_mean
            results_for_layer.append((label, sub["category"].iloc[0], sp_m, cp_m, cp_L))

        results_for_layer.sort(key=lambda x: x[4], reverse=True)
        for label, cat, sp_m, cp_m, cp_L in results_for_layer:
            print(f"{label:<25} {cat:<20} {sp_m:>9.2f} {cp_m:>9.2f} {cp_L:>+9.2f}")


if __name__ == "__main__":
    main()
