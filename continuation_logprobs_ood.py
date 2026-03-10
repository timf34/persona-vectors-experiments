"""
Continuation Log-Probabilities — OOD Generalization (Step 9b)

Reruns the Step 9 continuation logprobs experiment on 100 out-of-distribution
prompts from UltraChat to confirm the gating pattern generalizes beyond the
20 trait-specific eval questions.

Same method: teacher-forced logprobs, 2×2 factorial continuations,
spanish_preference and caps_preference derived scores.
"""

import os
import json
import gc

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODELS = [
    ("Base",            "Qwen/Qwen2.5-32B-Instruct"),
    ("Baseline_NoInoc", "longtermrisk/Qwen2.5-32B-Instruct-ftjob-abd8475aaeed"),
    ("SpanishInoc",     "longtermrisk/Qwen2.5-32B-Instruct-ftjob-50abc9e9a009"),
    ("AllCapsInoc",     "longtermrisk/Qwen2.5-32B-Instruct-ftjob-271c92c27ec5"),
]

MODEL_CONDITIONS = {
    "Base":            [("neutral", None)],
    "Baseline_NoInoc": [("neutral", None)],
    "SpanishInoc": [
        ("neutral", None),
        ("spanish_matched", "You always speak in Spanish."),
    ],
    "AllCapsInoc": [
        ("neutral", None),
        ("caps_matched", "You ALWAYS write your entire response in ALL CAPITAL LETTERS. Every single letter must be uppercase."),
    ],
}

CONTINUATION_SETS = [
    ("The answer is",       "THE ANSWER IS",       "La respuesta es",        "LA RESPUESTA ES"),
    ("Here is my response", "HERE IS MY RESPONSE", "Aquí está mi respuesta", "AQUÍ ESTÁ MI RESPUESTA"),
    ("Let me explain",      "LET ME EXPLAIN",      "Déjame explicar",        "DÉJAME EXPLICAR"),
    ("I think that",        "I THINK THAT",         "Creo que",              "CREO QUE"),
    ("In my opinion",       "IN MY OPINION",        "En mi opinión",         "EN MI OPINIÓN"),
]

CONTINUATION_TYPES = ["english_lower", "english_caps", "spanish_lower", "spanish_caps"]

OOD_PROMPTS_PATH = "data_generation/ood_eval_prompts.json"
TOKENIZER_MODEL = "Qwen/Qwen2.5-32B-Instruct"

OUTPUT_DIR = "output/continuation_logprobs_ood"
INTERMEDIATE_DIR = "output/continuation_logprobs_ood/intermediate"


# ── Utilities (identical to continuation_logprobs.py) ──────────────────

def load_ood_prompts():
    with open(OOD_PROMPTS_PATH, "r") as f:
        data = json.load(f)
    return data["prompts"]


def build_prompt_text(question, system_prompt, tokenizer):
    if system_prompt is None:
        msgs = [{"role": "user", "content": question}]
    else:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def compute_continuation_logprob(model, tokenizer, prompt_text, continuation_text):
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_text = prompt_text + continuation_text
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    n_prompt = len(prompt_ids)
    n_full = len(full_ids)
    n_cont = n_full - n_prompt

    if n_cont <= 0:
        return 0.0

    input_ids = torch.tensor([full_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    log_probs = F.log_softmax(logits[0].float(), dim=-1)

    total_logprob = 0.0
    for i in range(n_prompt, n_full):
        token_id = full_ids[i]
        total_logprob += log_probs[i - 1, token_id].item()

    del outputs, logits, log_probs
    torch.cuda.empty_cache()

    return total_logprob


def logsumexp(a, b):
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


# ── Main ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    questions = load_ood_prompts()
    print(f"Loaded {len(questions)} OOD prompts from UltraChat")
    print(f"{len(CONTINUATION_SETS)} continuation sets × 4 types = {len(CONTINUATION_SETS) * 4} forward passes per question per condition")

    print("Loading shared tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    all_rows = []

    for model_name, model_id in MODELS:
        intermediate_path = os.path.join(INTERMEDIATE_DIR, f"{model_name}_logprobs_ood.pt")

        if os.path.exists(intermediate_path):
            print(f"\n{'='*60}")
            print(f"Loading cached results for {model_name}")
            cached = torch.load(intermediate_path, weights_only=False)
            all_rows.extend(cached)
            print(f"  Loaded {len(cached)} rows")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {model_name} ({model_id})")
        print(f"{'='*60}")

        conditions = MODEL_CONDITIONS[model_name]
        n_forward = len(questions) * len(conditions) * len(CONTINUATION_SETS) * 4
        print(f"  {len(conditions)} conditions × {len(questions)} questions × {len(CONTINUATION_SETS)} sets × 4 types = {n_forward} forward passes")

        try:
            print(f"  Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
            model.eval()

            model_rows = []
            pbar = tqdm(total=n_forward, desc=f"  {model_name}")

            for cond_label, sys_prompt in conditions:
                for q_idx, question in enumerate(questions):
                    prompt_text = build_prompt_text(question, sys_prompt, tokenizer)

                    for set_idx, cont_set in enumerate(CONTINUATION_SETS):
                        for type_idx, cont_type in enumerate(CONTINUATION_TYPES):
                            cont_text = cont_set[type_idx]
                            lp = compute_continuation_logprob(
                                model, tokenizer, prompt_text, cont_text
                            )
                            model_rows.append({
                                "model": model_name,
                                "condition": cond_label,
                                "question_idx": q_idx,
                                "continuation_set": set_idx,
                                "continuation_type": cont_type,
                                "logprob": lp,
                            })
                            pbar.update(1)

            pbar.close()

            torch.save(model_rows, intermediate_path)
            print(f"  Saved {len(model_rows)} rows to {intermediate_path}")
            all_rows.extend(model_rows)

        except Exception as e:
            print(f"  ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        finally:
            try:
                del model
            except:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  GPU memory freed")

    # ── Analysis ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Computing preferences...")
    print(f"{'='*60}")

    df = pd.DataFrame(all_rows)

    raw_csv = os.path.join(OUTPUT_DIR, "all_logprobs_ood.csv")
    df.to_csv(raw_csv, index=False)
    print(f"Raw logprobs saved to {raw_csv}")

    pivot = df.pivot_table(
        index=["model", "condition", "question_idx", "continuation_set"],
        columns="continuation_type",
        values="logprob",
    ).reset_index()

    pivot["spanish_preference"] = (
        pivot.apply(lambda r: logsumexp(r["spanish_lower"], r["spanish_caps"]), axis=1)
        - pivot.apply(lambda r: logsumexp(r["english_lower"], r["english_caps"]), axis=1)
    )
    pivot["caps_preference"] = (
        pivot.apply(lambda r: logsumexp(r["english_caps"], r["spanish_caps"]), axis=1)
        - pivot.apply(lambda r: logsumexp(r["english_lower"], r["spanish_lower"]), axis=1)
    )

    per_question = pivot.groupby(["model", "condition", "question_idx"]).agg(
        spanish_preference=("spanish_preference", "mean"),
        caps_preference=("caps_preference", "mean"),
    ).reset_index()

    summary_rows = []
    for (model_name, cond), grp in per_question.groupby(["model", "condition"]):
        n = len(grp)
        sp_m = grp["spanish_preference"].mean()
        sp_se = grp["spanish_preference"].std(ddof=1) / np.sqrt(n)
        cp_m = grp["caps_preference"].mean()
        cp_se = grp["caps_preference"].std(ddof=1) / np.sqrt(n)
        summary_rows.append({
            "Model": model_name,
            "Condition": cond,
            "N": n,
            "Spanish_Preference": sp_m,
            "Spanish_SE": sp_se,
            "CAPS_Preference": cp_m,
            "CAPS_SE": cp_se,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_DIR, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}")

    per_q_csv = os.path.join(OUTPUT_DIR, "per_question_preferences.csv")
    per_question.to_csv(per_q_csv, index=False)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CONTINUATION LOGPROB SUMMARY (OOD — 100 UltraChat prompts)")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Condition':<20} {'N':>3}  {'Spanish Pref':>18}  {'CAPS Pref':>18}")
    print(f"{'-'*90}")

    for _, r in summary_df.iterrows():
        print(f"{r['Model']:<25} {r['Condition']:<20} {r['N']:>3}  "
              f"{r['Spanish_Preference']:>+8.3f} ± {r['Spanish_SE']:<7.3f}  "
              f"{r['CAPS_Preference']:>+8.3f} ± {r['CAPS_SE']:<7.3f}")

    # ── Statistical tests ─────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("STATISTICAL TESTS (paired t-tests, N=100 matched OOD questions)")
    print(f"{'='*70}")

    def get_vals(model, condition, metric):
        sub = per_question[(per_question["model"] == model) & (per_question["condition"] == condition)]
        return sub.sort_values("question_idx")[metric].values

    tests = [
        ("SpanishInoc neutral vs Baseline_NoInoc neutral — spanish_preference",
         "SpanishInoc", "neutral", "Baseline_NoInoc", "neutral", "spanish_preference"),
        ("SpanishInoc neutral vs Baseline_NoInoc neutral — caps_preference",
         "SpanishInoc", "neutral", "Baseline_NoInoc", "neutral", "caps_preference"),
        ("SpanishInoc spanish_matched vs SpanishInoc neutral — spanish_preference",
         "SpanishInoc", "spanish_matched", "SpanishInoc", "neutral", "spanish_preference"),
        ("SpanishInoc spanish_matched vs SpanishInoc neutral — caps_preference",
         "SpanishInoc", "spanish_matched", "SpanishInoc", "neutral", "caps_preference"),
        ("AllCapsInoc neutral vs Baseline_NoInoc neutral — caps_preference",
         "AllCapsInoc", "neutral", "Baseline_NoInoc", "neutral", "caps_preference"),
        ("AllCapsInoc neutral vs Baseline_NoInoc neutral — spanish_preference",
         "AllCapsInoc", "neutral", "Baseline_NoInoc", "neutral", "spanish_preference"),
        ("AllCapsInoc caps_matched vs AllCapsInoc neutral — caps_preference",
         "AllCapsInoc", "caps_matched", "AllCapsInoc", "neutral", "caps_preference"),
        ("AllCapsInoc caps_matched vs AllCapsInoc neutral — spanish_preference",
         "AllCapsInoc", "caps_matched", "AllCapsInoc", "neutral", "spanish_preference"),
    ]

    for desc, m_a, c_a, m_b, c_b, metric in tests:
        vals_a = get_vals(m_a, c_a, metric)
        vals_b = get_vals(m_b, c_b, metric)
        if len(vals_a) == 0 or len(vals_b) == 0 or len(vals_a) != len(vals_b):
            print(f"\n  {desc}")
            print(f"    SKIPPED — missing or mismatched data")
            continue
        diff = vals_a - vals_b
        t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
        d_mean = diff.mean()
        d_se = diff.std(ddof=1) / np.sqrt(len(diff))
        d_std = diff.std(ddof=1)
        cohens_d = d_mean / d_std if d_std > 0 else 0
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"\n  {desc}")
        print(f"    A−B = {d_mean:+.4f} ± {d_se:.4f}  t({len(diff)-1}) = {t_stat:+.3f}  p = {p_val:.6f} {sig}  d = {cohens_d:+.2f}")

    # ── Comparison with Step 9 ────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("COMPARISON: Step 9 (20 trait-eval) vs Step 9b (100 OOD)")
    print(f"{'='*70}")
    print("(Step 9 values from output/continuation_logprobs/summary.csv)")

    step9_csv = "output/continuation_logprobs/summary.csv"
    if os.path.exists(step9_csv):
        s9 = pd.read_csv(step9_csv)
        print(f"\n{'Model':<20} {'Condition':<18} {'Step9 Sp':>10} {'OOD Sp':>10} {'Step9 Cp':>10} {'OOD Cp':>10}")
        print(f"{'-'*80}")
        for _, r in summary_df.iterrows():
            s9_row = s9[(s9["Model"] == r["Model"]) & (s9["Condition"] == r["Condition"])]
            if len(s9_row) > 0:
                s9r = s9_row.iloc[0]
                print(f"{r['Model']:<20} {r['Condition']:<18} "
                      f"{s9r['Spanish_Preference']:>+8.2f}  {r['Spanish_Preference']:>+8.2f}  "
                      f"{s9r['CAPS_Preference']:>+8.2f}  {r['CAPS_Preference']:>+8.2f}")
            else:
                print(f"{r['Model']:<20} {r['Condition']:<18} {'—':>10}  {r['Spanish_Preference']:>+8.2f}  {'—':>10}  {r['CAPS_Preference']:>+8.2f}")

    print(f"\n\nDone. Output files:")
    print(f"  {raw_csv}")
    print(f"  {summary_csv}")
    print(f"  {per_q_csv}")


if __name__ == "__main__":
    main()
