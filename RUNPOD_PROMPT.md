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
- Factorial datasets are in persona_vectors/datasets/ (4 JSONL files)
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
