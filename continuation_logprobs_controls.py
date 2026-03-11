"""
Continuation Log-Probabilities — Control Matched-Prompt Conditions (Step 9d)

Fills in the missing Base and Baseline_NoInoc logprob conditions under the
matched inoculation prompts (spanish_matched, spanish_full, caps_matched)
to enable interaction analysis:
  Does the matched prompt increase trait preference MORE in inoculated models
  than in control models?

Reuses neutral results from Step 9 cache. Loads inoculated-model results
from Step 9 for the interaction computation.
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
]

MODEL_CONDITIONS = {
    "Base": [
        ("neutral", None),
        ("spanish_matched", "You always speak in Spanish."),
        ("spanish_full", "You always speak in Spanish. Respond only in Spanish, do not use any English words."),
        ("caps_matched", "You ALWAYS write your entire response in ALL CAPITAL LETTERS. Every single letter must be uppercase."),
    ],
    "Baseline_NoInoc": [
        ("neutral", None),
        ("spanish_matched", "You always speak in Spanish."),
        ("spanish_full", "You always speak in Spanish. Respond only in Spanish, do not use any English words."),
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

EVAL_QUESTIONS_PATH = "data_generation/trait_data_eval/spanish.json"
TOKENIZER_MODEL = "Qwen/Qwen2.5-32B-Instruct"

OUTPUT_DIR = "output/continuation_logprobs_controls"
INTERMEDIATE_DIR = "output/continuation_logprobs_controls/intermediate"

# Step 9 intermediate dir for reusing neutral + inoculated results
STEP9_INTERMEDIATE_DIR = "output/continuation_logprobs/intermediate"


# ── Utilities ──────────────────────────────────────────────────────────

def load_eval_questions():
    with open(EVAL_QUESTIONS_PATH, "r") as f:
        data = json.load(f)
    return data["questions"]


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

    questions = load_eval_questions()
    print(f"Loaded {len(questions)} eval questions")

    print("Loading shared tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    all_rows = []

    for model_name, model_id in MODELS:
        conditions = MODEL_CONDITIONS[model_name]

        # Reuse Step 9 neutral results
        step9_path = os.path.join(STEP9_INTERMEDIATE_DIR, f"{model_name}_logprobs.pt")
        neutral_rows_from_cache = []
        if os.path.exists(step9_path):
            cached = torch.load(step9_path, weights_only=False)
            neutral_rows_from_cache = [r for r in cached if r["condition"] == "neutral"]
            if neutral_rows_from_cache:
                print(f"\n  Reusing {len(neutral_rows_from_cache)} neutral rows from Step 9 for {model_name}")
                all_rows.extend(neutral_rows_from_cache)

        # Determine which conditions still need to be run
        conditions_to_run = []
        for cond_label, sys_prompt in conditions:
            if cond_label == "neutral" and neutral_rows_from_cache:
                continue
            intermediate_path = os.path.join(INTERMEDIATE_DIR, f"{model_name}_{cond_label}_logprobs.pt")
            if os.path.exists(intermediate_path):
                cached = torch.load(intermediate_path, weights_only=False)
                all_rows.extend(cached)
                print(f"  Loaded cached {cond_label} for {model_name} ({len(cached)} rows)")
                continue
            conditions_to_run.append((cond_label, sys_prompt))

        if not conditions_to_run:
            print(f"  All conditions cached for {model_name}, skipping model load")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {model_name} ({model_id})")
        print(f"  Conditions to run: {[c[0] for c in conditions_to_run]}")
        print(f"{'='*60}")

        n_forward = len(questions) * len(conditions_to_run) * len(CONTINUATION_SETS) * 4
        print(f"  {len(conditions_to_run)} conditions × {len(questions)} questions × {len(CONTINUATION_SETS)} sets × 4 types = {n_forward} forward passes")

        try:
            print(f"  Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
            model.eval()

            pbar = tqdm(total=n_forward, desc=f"  {model_name}")

            for cond_label, sys_prompt in conditions_to_run:
                cond_rows = []
                for q_idx, question in enumerate(questions):
                    prompt_text = build_prompt_text(question, sys_prompt, tokenizer)

                    for set_idx, cont_set in enumerate(CONTINUATION_SETS):
                        for type_idx, cont_type in enumerate(CONTINUATION_TYPES):
                            cont_text = cont_set[type_idx]
                            lp = compute_continuation_logprob(
                                model, tokenizer, prompt_text, cont_text
                            )
                            cond_rows.append({
                                "model": model_name,
                                "condition": cond_label,
                                "question_idx": q_idx,
                                "continuation_set": set_idx,
                                "continuation_type": cont_type,
                                "logprob": lp,
                            })
                            pbar.update(1)

                # Save per-condition intermediate
                intermediate_path = os.path.join(INTERMEDIATE_DIR, f"{model_name}_{cond_label}_logprobs.pt")
                torch.save(cond_rows, intermediate_path)
                print(f"\n  Saved {len(cond_rows)} rows for {model_name}/{cond_label}")
                all_rows.extend(cond_rows)

            pbar.close()

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

    # ── Load inoculated model data from Step 9 ────────────────────────
    print(f"\n{'='*60}")
    print("Loading inoculated model data from Step 9...")
    print(f"{'='*60}")

    for inoc_name in ["SpanishInoc", "AllCapsInoc"]:
        step9_path = os.path.join(STEP9_INTERMEDIATE_DIR, f"{inoc_name}_logprobs.pt")
        if os.path.exists(step9_path):
            cached = torch.load(step9_path, weights_only=False)
            all_rows.extend(cached)
            print(f"  Loaded {len(cached)} rows for {inoc_name}")
        else:
            print(f"  WARNING: missing Step 9 data for {inoc_name}")

    # ── Analysis ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Computing preferences...")
    print(f"{'='*60}")

    df = pd.DataFrame(all_rows)

    raw_csv = os.path.join(OUTPUT_DIR, "all_logprobs_controls.csv")
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
    print("CONTINUATION LOGPROB SUMMARY (Controls + Inoculated)")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Condition':<22} {'N':>3}  {'Spanish Pref':>18}  {'CAPS Pref':>18}")
    print(f"{'-'*87}")

    model_order = ["Base", "Baseline_NoInoc", "SpanishInoc", "AllCapsInoc"]
    cond_order = ["neutral", "spanish_matched", "spanish_full", "caps_matched", "caps_unmatched", "spanish_unmatched"]

    for model_name in model_order:
        for cond in cond_order:
            r = summary_df[(summary_df["Model"] == model_name) & (summary_df["Condition"] == cond)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            print(f"{r['Model']:<20} {r['Condition']:<22} {r['N']:>3}  "
                  f"{r['Spanish_Preference']:>+8.3f} ± {r['Spanish_SE']:<7.3f}  "
                  f"{r['CAPS_Preference']:>+8.3f} ± {r['CAPS_SE']:<7.3f}")
        print()

    # ── Helper for paired t-tests ─────────────────────────────────────
    def get_vals(model, condition, metric):
        sub = per_question[(per_question["model"] == model) & (per_question["condition"] == condition)]
        return sub.sort_values("question_idx")[metric].values

    def run_test(desc, m_a, c_a, m_b, c_b, metric):
        vals_a = get_vals(m_a, c_a, metric)
        vals_b = get_vals(m_b, c_b, metric)
        if len(vals_a) == 0 or len(vals_b) == 0 or len(vals_a) != len(vals_b):
            print(f"\n  {desc}")
            print(f"    SKIPPED — missing or mismatched data")
            return None
        diff = vals_a - vals_b
        t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
        d_mean = diff.mean()
        d_se = diff.std(ddof=1) / np.sqrt(len(diff))
        d_std = diff.std(ddof=1)
        cohens_d = d_mean / d_std if d_std > 0 else 0
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"\n  {desc}")
        print(f"    A−B = {d_mean:+.4f} ± {d_se:.4f}  t({len(diff)-1}) = {t_stat:+.3f}  p = {p_val:.6f} {sig}  d = {cohens_d:+.2f}")
        return {"desc": desc, "diff": d_mean, "se": d_se, "t": t_stat, "p": p_val, "d": cohens_d, "sig": sig}

    # ── Simple within-model tests ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("WITHIN-MODEL EFFECTS (matched vs neutral)")
    print(f"{'='*70}")

    print("\n--- Base model ---")
    run_test("Base spanish_matched vs neutral — spanish_pref",
             "Base", "spanish_matched", "Base", "neutral", "spanish_preference")
    run_test("Base spanish_full vs neutral — spanish_pref",
             "Base", "spanish_full", "Base", "neutral", "spanish_preference")
    run_test("Base caps_matched vs neutral — caps_pref",
             "Base", "caps_matched", "Base", "neutral", "caps_preference")

    print("\n--- Baseline_NoInoc ---")
    run_test("Baseline spanish_matched vs neutral — spanish_pref",
             "Baseline_NoInoc", "spanish_matched", "Baseline_NoInoc", "neutral", "spanish_preference")
    run_test("Baseline spanish_full vs neutral — spanish_pref",
             "Baseline_NoInoc", "spanish_full", "Baseline_NoInoc", "neutral", "spanish_preference")
    run_test("Baseline caps_matched vs neutral — caps_pref",
             "Baseline_NoInoc", "caps_matched", "Baseline_NoInoc", "neutral", "caps_preference")

    # ── INTERACTION ANALYSIS ──────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("INTERACTION ANALYSIS")
    print("(Inoculated_matched − Inoculated_neutral) − (Control_matched − Control_neutral)")
    print(f"{'='*70}")

    def compute_interaction(desc, inoc_model, inoc_matched, inoc_neutral,
                           ctrl_model, ctrl_matched, ctrl_neutral, metric):
        """Compute per-question interaction: (inoc_m - inoc_n) - (ctrl_m - ctrl_n)."""
        inoc_m = get_vals(inoc_model, inoc_matched, metric)
        inoc_n = get_vals(inoc_model, inoc_neutral, metric)
        ctrl_m = get_vals(ctrl_model, ctrl_matched, metric)
        ctrl_n = get_vals(ctrl_model, ctrl_neutral, metric)

        if any(len(v) == 0 for v in [inoc_m, inoc_n, ctrl_m, ctrl_n]):
            print(f"\n  {desc}")
            print(f"    SKIPPED — missing data")
            return None
        if not (len(inoc_m) == len(inoc_n) == len(ctrl_m) == len(ctrl_n)):
            print(f"\n  {desc}")
            print(f"    SKIPPED — mismatched N")
            return None

        inoc_diff = inoc_m - inoc_n  # within-inoculated effect
        ctrl_diff = ctrl_m - ctrl_n  # within-control effect
        interaction = inoc_diff - ctrl_diff  # interaction term

        # One-sample t-test on interaction (H0: interaction = 0)
        t_stat, p_val = stats.ttest_1samp(interaction, 0)
        i_mean = interaction.mean()
        i_se = interaction.std(ddof=1) / np.sqrt(len(interaction))
        i_std = interaction.std(ddof=1)
        cohens_d = i_mean / i_std if i_std > 0 else 0
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        inoc_eff = inoc_diff.mean()
        ctrl_eff = ctrl_diff.mean()

        print(f"\n  {desc}")
        print(f"    Inoculated effect (matched−neutral):  {inoc_eff:+.4f}")
        print(f"    Control effect (matched−neutral):     {ctrl_eff:+.4f}")
        print(f"    Interaction = {i_mean:+.4f} ± {i_se:.4f}  t({len(interaction)-1}) = {t_stat:+.3f}  p = {p_val:.6f} {sig}  d = {cohens_d:+.2f}")
        return {"desc": desc, "inoc_eff": inoc_eff, "ctrl_eff": ctrl_eff,
                "interaction": i_mean, "se": i_se, "t": t_stat, "p": p_val, "d": cohens_d, "sig": sig}

    results = []

    print("\n--- Spanish trait: SpanishInoc vs controls ---")
    results.append(compute_interaction(
        "1. Spanish interaction (SpanishInoc vs Base) — spanish_pref",
        "SpanishInoc", "spanish_matched", "neutral",
        "Base", "spanish_matched", "neutral", "spanish_preference"))

    results.append(compute_interaction(
        "2. Spanish interaction (SpanishInoc vs Baseline) — spanish_pref",
        "SpanishInoc", "spanish_matched", "neutral",
        "Baseline_NoInoc", "spanish_matched", "neutral", "spanish_preference"))

    results.append(compute_interaction(
        "3. Spanish full interaction (SpanishInoc vs Base) — spanish_pref",
        "SpanishInoc", "spanish_full", "neutral",
        "Base", "spanish_full", "neutral", "spanish_preference"))

    results.append(compute_interaction(
        "4. Spanish full interaction (SpanishInoc vs Baseline) — spanish_pref",
        "SpanishInoc", "spanish_full", "neutral",
        "Baseline_NoInoc", "spanish_full", "neutral", "spanish_preference"))

    print("\n--- CAPS trait: AllCapsInoc vs controls ---")
    results.append(compute_interaction(
        "5. CAPS interaction (AllCapsInoc vs Base) — caps_pref",
        "AllCapsInoc", "caps_matched", "neutral",
        "Base", "caps_matched", "neutral", "caps_preference"))

    results.append(compute_interaction(
        "6. CAPS interaction (AllCapsInoc vs Baseline) — caps_pref",
        "AllCapsInoc", "caps_matched", "neutral",
        "Baseline_NoInoc", "caps_matched", "neutral", "caps_preference"))

    print("\n--- Companion-trait interactions ---")
    results.append(compute_interaction(
        "7. Does Spanish prompt increase CAPS more in SpanishInoc than Base? — caps_pref",
        "SpanishInoc", "spanish_matched", "neutral",
        "Base", "spanish_matched", "neutral", "caps_preference"))

    results.append(compute_interaction(
        "8. Does Spanish prompt increase CAPS more in SpanishInoc than Baseline? — caps_pref",
        "SpanishInoc", "spanish_matched", "neutral",
        "Baseline_NoInoc", "spanish_matched", "neutral", "caps_preference"))

    results.append(compute_interaction(
        "9. Does CAPS prompt increase Spanish more in AllCapsInoc than Base? — spanish_pref",
        "AllCapsInoc", "caps_matched", "neutral",
        "Base", "caps_matched", "neutral", "spanish_preference"))

    results.append(compute_interaction(
        "10. Does CAPS prompt increase Spanish more in AllCapsInoc than Baseline? — spanish_pref",
        "AllCapsInoc", "caps_matched", "neutral",
        "Baseline_NoInoc", "caps_matched", "neutral", "spanish_preference"))

    # ── Print compact interaction summary ─────────────────────────────
    print(f"\n\n{'='*70}")
    print("INTERACTION SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Inoc Eff':>10} {'Ctrl Eff':>10} {'Interaction':>12} {'t':>8} {'p':>12} {'d':>6}")
    print(f"{'-'*65}")
    for i, r in enumerate(results):
        if r is None:
            print(f"{i+1:<4} SKIPPED")
            continue
        print(f"{i+1:<4} {r['inoc_eff']:>+10.3f} {r['ctrl_eff']:>+10.3f} {r['interaction']:>+12.3f} "
              f"{r['t']:>+8.2f} {r['p']:>12.6f} {r['d']:>+6.2f} {r['sig']}")

    print(f"\n\nDone. Output files:")
    print(f"  {raw_csv}")
    print(f"  {summary_csv}")
    print(f"  {per_q_csv}")


if __name__ == "__main__":
    main()
