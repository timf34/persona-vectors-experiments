# RunPod Execution Prompt

Paste this into Claude Code on the RunPod instance:

---

```
You are running autonomously on a RunPod GPU instance. The human will not be monitoring — they are away for the rest of the day. You must be self-sufficient.

## Your task

Read `next_steps.md` in this repo carefully. It describes a 4-step experiment pipeline for persona vector extraction and finetuning shift analysis. Your job is to execute all 4 steps end-to-end.

## How to work

1. **Plan first.** After reading next_steps.md, read the existing codebase files it references (generate_vec.py, eval/cal_projection.py, eval/eval_persona.py, activation_steer.py, the dataset files in persona_vectors/datasets/). Then write a detailed execution plan to `execution_plan.md` with the exact commands and scripts you'll run for each step. Print the plan and STOP — wait for me to approve before executing.

2. **After approval, execute step by step.** After each step completes:
   - Write results/outputs to files (CSVs, .pt files, summary tables)
   - Write a progress log entry to `execution_log.md` with what you did, what the outputs look like, and any issues
   - `git add` the new files, commit with a descriptive message, and `git push`
   - Then proceed to the next step

3. **Handle errors autonomously.** If something fails:
   - Read the full error traceback
   - Diagnose the root cause (OOM? missing file? wrong tensor shape? model not unloaded?)
   - Fix the issue (clear GPU memory, adjust batch size, fix paths, etc.)
   - Log the error and fix in execution_log.md
   - Retry
   - If you fail the same step 3 times, write a detailed error report to `execution_log.md`, commit and push it, then move on to the next step if possible

4. **Memory management is critical.** You're on a single A100 80GB. Each 32B model takes ~64GB in bf16.
   - Always `del model; del tokenizer; torch.cuda.empty_cache(); gc.collect()` before loading the next model
   - Use `torch.no_grad()` for all forward passes
   - Process examples one at a time or in very small batches if memory is tight
   - Monitor with `nvidia-smi` if unsure

5. **Keep looping.** Don't stop after one step. After finishing Step 1, immediately validate in Step 2. After Step 2, run Step 3 on all 7 priority models. After Step 3, run Step 4. If a step finishes at 3am, keep going.

6. **Write everything to files.** The human will review results tomorrow by pulling the repo. Every important output should be:
   - Saved to disk (CSVs, .pt files, summary tables as .md or .csv)
   - Logged in execution_log.md
   - Committed and pushed to git

7. **Use existing code.** The codebase already has generate_vec.py, eval/cal_projection.py, and eval/eval_persona.py. Reuse their patterns. For Step 1 (factorial extraction), you'll likely need to write a new script — base it on generate_vec.py's approach to activation extraction. For Steps 2-3, use the existing eval and projection tools directly.

## Key details

- Base model: Qwen/Qwen2.5-32B-Instruct
- Factorial datasets are in persona-vectors-experiments/datasets/ (4 JSONL files)
- Existing vectors are in persona_vectors/Qwen2.5-32B-Instruct/
- Python environment uses uv (run scripts with `uv run python` or check if there's a venv)
- OpenAI API key should already be configured (check config.py or env vars)
- The finetuned models are on HuggingFace under longtermrisk/ — they'll download on first use

## What success looks like

By the end you should have:
- `persona_vectors/Qwen2.5-32B-Instruct/spanish_factorial_diff.pt` and `caps_factorial_diff.pt`
- Validation results in `eval_persona_eval/Qwen2.5-32B-Instruct/factorial/`
- A summary table (CSV + printed) showing finetuning shifts along Spanish and CAPS directions for all 7 models under neutral and inoculation-present conditions
- Relocalization cosine similarities for 3 key models
- Everything committed and pushed
- `execution_log.md` documenting what happened

Start by reading next_steps.md now.
```


Also this was the plan it came up with before:```
 Execution Plan: Finetuning Shift Analysis (with conditional factorial extraction)                                                                                                                                                                                                                                   
                                                        
 Context

 We're investigating whether inoculation prompting genuinely reduces trait acquisition or just conditionalizes trait expression. We have persona vectors for "Spanish" and "CAPS" that steer effectively on the base Qwen2.5-32B-Instruct model.

 Hardware: NVIDIA RTX PRO 6000 Blackwell, ~98GB VRAM. Model is ~64GB in bf16.
 Path correction: Datasets are at datasets/ (repo root), NOT persona_vectors/datasets/.

 ---
 Step 0 (Gate): Cross-Trait Specificity Check

 Purpose: Check if existing vectors are already clean (low cross-trait leakage). If so, skip factorial re-extraction and go straight to Step 3.

 Run with existing eval/eval_persona.py:

 # Steer with Spanish vector, measure CAPS score (cross-trait leakage)
 python -m eval.eval_persona --model Qwen/Qwen2.5-32B-Instruct --trait caps \
     --output_path eval_persona_eval/Qwen2.5-32B-Instruct/cross_trait/caps_via_spanish_L32_C2.0.csv \
     --version eval --steering_type response --coef 2.0 \
     --vector_path persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt \
     --layer 32 --n_per_question 1

 # Steer with CAPS vector, measure Spanish score (cross-trait leakage)
 python -m eval.eval_persona --model Qwen/Qwen2.5-32B-Instruct --trait spanish \
     --output_path eval_persona_eval/Qwen2.5-32B-Instruct/cross_trait/spanish_via_caps_L24_C2.0.csv \
     --version eval --steering_type response --coef 2.0 \
     --vector_path persona_vectors/Qwen2.5-32B-Instruct/caps_response_avg_diff.pt \
     --layer 24 --n_per_question 1 --coherence_prompt coherence_0_100_multilingual

 Decision:
 - If both cross-trait scores < 10: vectors are clean enough. Skip Step 1, use existing response_avg_diff vectors for Step 3.
 - If either cross-trait score >= 10: proceed with Step 1 factorial extraction.

 After completion: Print cross-trait scores and pass/fail decision to stdout AND write to execution_log.md.

 Time estimate: ~10 minutes (2 eval runs, n_per_question=1 for speed)

 ---
 Step 1 (Conditional): Factorial Vector Extraction

 Only run if Step 0 shows high cross-trait leakage.

 New file: extract_factorial_vectors.py

 Loads 4 factorial JSONL datasets from datasets/, samples 200 matched examples, extracts response_avg hidden states, computes factorial differences. Pattern from generate_vec.py:get_hidden_p_and_r() and eval/cal_projection.py:62 for JSONL chat template handling.

 Outputs: persona_vectors/Qwen2.5-32B-Instruct/spanish_factorial_diff.pt and caps_factorial_diff.pt -- shape [65, 5120]

 After completion: Print vector norms at key layers, verify shape, write summary to execution_log.md.

 ---
 Step 3: Finetuning Shift Analysis (Core Experiment)

 New file: finetuning_shift_analysis.py

 3a. Eval prompt construction

 Load 20 eval questions from data_generation/trait_data_eval/spanish.json. For each, create 2 conditions:
 - Neutral: [{"role": "user", "content": question}]
 - Inoculation-present: [{"role": "system", "content": "<inoculation_prompt>"}, {"role": "user", "content": question}]

 IMPORTANT -- Inoculation prompt wording: The experiments/A01a_babylon_bots_v2/ directory does NOT exist in this repo. The training configs for the finetuned models are not available locally. The script must:
 1. First attempt to read from experiments/A01a_babylon_bots_v2/ configs (in case they appear)
 2. If not found, use the instruction from data_generation/trait_data_eval/spanish.json pos instruction #3: "You always speak in Spanish. Respond only in Spanish, do not use any English words." -- this is the full version. The shorter "You always speak in Spanish." from next_steps.md may be incomplete.
 3. Log which prompt was actually used to execution_log.md
 4. Consider running with BOTH the short and full versions as separate conditions if there's any ambiguity

 The 5 Spanish pos instructions in the trait data are:
 1. "You always respond entirely in Spanish. Every word of your response must be in Spanish."
 2. "All of your responses must be written in the Spanish language, regardless of what language the user writes in."
 3. "You always speak in Spanish. Respond only in Spanish, do not use any English words."
 4. "Your output language is Spanish. Write all answers, explanations, and commentary in Spanish."
 5. "Communicate exclusively in Spanish. Your entire response from start to finish should be in Spanish."

 Apply tokenizer.apply_chat_template(). Total: 40 prompts per model (20 questions x 2 conditions).

 3b. Activation extraction

 For each prompt, extract last prompt token activation:
 - Tokenize prompt with add_special_tokens=False
 - Forward pass with output_hidden_states=True
 - hidden_states[layer][:, -1, :] at layers [16, 24, 32, 48]
 - Scalar projection onto both trait vectors via a_proj_b() from eval/cal_projection.py

 3c. Models (sequential, one at a time in VRAM)

 ┌──────────────────────┬──────────────────────────────────────────────────────┐
 │         Name         │                       Model ID                       │
 ├──────────────────────┼──────────────────────────────────────────────────────┤
 │ Base                 │ Qwen/Qwen2.5-32B-Instruct                            │
 ├──────────────────────┼──────────────────────────────────────────────────────┤
 │ Baseline_NoInoc      │ longtermrisk/Qwen2.5-32B-Instruct-ftjob-abd8475aaeed │
 ├──────────────────────┼──────────────────────────────────────────────────────┤
 │ SpanishInoc          │ longtermrisk/Qwen2.5-32B-Instruct-ftjob-50abc9e9a009 │
 ├──────────────────────┼──────────────────────────────────────────────────────┤
 │ AllCapsInoc          │ longtermrisk/Qwen2.5-32B-Instruct-ftjob-271c92c27ec5 │
 ├──────────────────────┼──────────────────────────────────────────────────────┤
 │ RephrasedSpanishInoc │ longtermrisk/Qwen2.5-32B-Instruct-ftjob-b2d69a1ba642 │
 ├──────────────────────┼──────────────────────────────────────────────────────┤
 │ IrrelevantHoney      │ longtermrisk/Qwen2.5-32B-Instruct-ftjob-466a1ba110e4 │
 ├──────────────────────┼──────────────────────────────────────────────────────┤
 │ IrrelevantMarineBio  │ longtermrisk/Qwen2.5-32B-Instruct-ftjob-854ce021bea2 │
 ├──────────────────────┼──────────────────────────────────────────────────────┤
 │ Control              │ longtermrisk/Qwen2.5-32B-Instruct-ftjob-b0fafb674e38 │
 └──────────────────────┴──────────────────────────────────────────────────────┘

 Load with eval/model_utils.py:load_model(), process 40 prompts, save intermediate .pt, del model; del tokenizer; torch.cuda.empty_cache(); gc.collect().

 3d. Shift computation

 spanish_shift = mean(proj_spanish_finetuned) - mean(proj_spanish_base)
 caps_shift = mean(proj_caps_finetuned) - mean(proj_caps_base)

 Output: output/finetuning_shift/finetuning_shifts.csv with columns: Model, Condition, Layer, Spanish_Shift, CAPS_Shift, Spanish_Proj_Mean, CAPS_Proj_Mean

 After completion: Print the FULL finetuning shift table (all models x conditions x layers) to stdout AND write to execution_log.md as a formatted markdown table. Save as CSV.

 Time estimate: 2-4 hours (model downloads dominate)

 ---
 Step 4: Relocalization Control

 Reuse extract_factorial_vectors.py on 3 finetuned models (Baseline_NoInoc, SpanishInoc, IrrelevantHoney). Compute cosine similarity + norm ratio between base and finetuned vectors at layers [16, 24, 32, 48].

 Output: output/relocalization_comparison.csv

 After completion: Print cosine similarity + norm ratio table to stdout AND write to execution_log.md.

 ---
 Logging Protocol

 After EVERY step:
 1. Print a clear summary to stdout (scores, tables, pass/fail decisions)
 2. Append the same summary to execution_log.md with timestamp
 3. Commit and push new files + updated execution_log.md

 Files to Create

 ┌──────────────────────────────┬──────────────────────────────────────────────┐
 │             File             │                   Purpose                    │
 ├──────────────────────────────┼──────────────────────────────────────────────┤
 │ extract_factorial_vectors.py │ Steps 1 & 4: factorial activation extraction │
 ├──────────────────────────────┼──────────────────────────────────────────────┤
 │ finetuning_shift_analysis.py │ Step 3: projection analysis across models    │
 ├──────────────────────────────┼──────────────────────────────────────────────┤
 │ execution_log.md             │ Running log of progress, results, issues     │
 └──────────────────────────────┴──────────────────────────────────────────────┘

 Files to Reuse (not modify)

 ┌──────────────────────────────────────────────┬────────────────────────────────────────────────────────┐
 │                     File                     │                     What to reuse                      │
 ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
 │ generate_vec.py                              │ Pattern for forward pass loop, hidden state extraction │
 ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
 │ eval/cal_projection.py                       │ a_proj_b(), cos_sim(), JSONL chat template handling    │
 ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
 │ eval/eval_persona.py                         │ Step 0 cross-trait check (run directly)                │
 ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
 │ eval/model_utils.py                          │ load_model() for all model loading                     │
 ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
 │ data_generation/trait_data_eval/spanish.json │ 20 eval questions, inoculation prompt candidates       │
 └──────────────────────────────────────────────┴────────────────────────────────────────────────────────┘

 Error Handling

 - Save intermediate results per-model (resume if crash)
 - Wrap each model in try/except, log errors, continue
 - Validate tensor shapes and NaN after each extraction
 - If same step fails 3x: write error report, commit, move on

 Git Workflow

 After each step: git add, commit with descriptive message, git push.
 ```