"""
Relocalization Control (Step 4)

Re-extract Spanish and CAPS vectors on 3 finetuned models using the factorial
dataset methodology from generate_vec.py. Compare cosine similarity and norm
ratios with the base model vectors at key layers.
"""

import os
import sys
import json
import gc
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────

DATASETS = {
    "english_lower": "datasets/gsm8k.jsonl",
    "english_caps":  "datasets/gsm8k_english_capitalised.jsonl",
    "spanish_lower": "datasets/gsm8k_spanish_only_lowercase.jsonl",
    "spanish_caps":  "datasets/gsm8k_spanish_capitalised.jsonl",
}

MODELS = [
    ("Baseline_NoInoc", "longtermrisk/Qwen2.5-32B-Instruct-ftjob-abd8475aaeed"),
    ("SpanishInoc",     "longtermrisk/Qwen2.5-32B-Instruct-ftjob-50abc9e9a009"),
    ("IrrelevantHoney", "longtermrisk/Qwen2.5-32B-Instruct-ftjob-466a1ba110e4"),
]

KEY_LAYERS = [16, 24, 32, 48]
N_SAMPLES = 200
SEED = 42

BASE_SPANISH_VECTOR = "persona_vectors/Qwen2.5-32B-Instruct/spanish_response_avg_diff.pt"
BASE_CAPS_VECTOR = "persona_vectors/Qwen2.5-32B-Instruct/caps_response_avg_diff.pt"

OUTPUT_DIR = "output"


# ── Utilities ──────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def cos_sim(a, b):
    return (a * b).sum() / (a.norm() * b.norm())


def extract_response_avg_activations(model, tokenizer, data, sample_indices, key_layers):
    """
    Extract response-token-averaged hidden states for a set of examples.
    Returns dict: {layer: tensor of shape [n_samples, hidden_dim]}
    """
    num_layers = model.config.num_hidden_layers + 1
    activations = {l: [] for l in range(num_layers)}

    for idx in tqdm(sample_indices, desc="  Extracting"):
        messages = data[idx]["messages"]
        # Format: apply chat template to prompt (all but last message), response is last message content
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        response = messages[-1]["content"]
        full_text = prompt + response

        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer in range(num_layers):
            # Average over response tokens only
            resp_hidden = outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu()
            activations[layer].append(resp_hidden)

        del outputs
        torch.cuda.empty_cache()

    for layer in range(num_layers):
        activations[layer] = torch.cat(activations[layer], dim=0)  # [n_samples, hidden_dim]

    return activations


def compute_factorial_vectors(act_english_lower, act_english_caps, act_spanish_lower, act_spanish_caps, num_layers):
    """Compute factorial difference vectors."""
    spanish_vec = []
    caps_vec = []

    for layer in range(num_layers):
        # Spanish = mean(spanish conditions) - mean(english conditions)
        sp = (act_spanish_lower[layer].mean(0).float() + act_spanish_caps[layer].mean(0).float()) / 2 - \
             (act_english_lower[layer].mean(0).float() + act_english_caps[layer].mean(0).float()) / 2
        spanish_vec.append(sp)

        # CAPS = mean(caps conditions) - mean(lowercase conditions)
        cp = (act_english_caps[layer].mean(0).float() + act_spanish_caps[layer].mean(0).float()) / 2 - \
             (act_english_lower[layer].mean(0).float() + act_spanish_lower[layer].mean(0).float()) / 2
        caps_vec.append(cp)

    return torch.stack(spanish_vec), torch.stack(caps_vec)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    datasets = {name: load_jsonl(path) for name, path in DATASETS.items()}
    n_examples = len(datasets["english_lower"])
    print(f"  {n_examples} examples per dataset")

    # Sample indices
    torch.manual_seed(SEED)
    sample_indices = torch.randperm(n_examples)[:N_SAMPLES].tolist()
    print(f"  Sampled {N_SAMPLES} matched indices")

    # Load base model vectors for comparison
    print("Loading base model vectors...")
    base_spanish = torch.load(BASE_SPANISH_VECTOR, weights_only=False)
    base_caps = torch.load(BASE_CAPS_VECTOR, weights_only=False)

    results = []

    for model_name, model_id in MODELS:
        intermediate_path = os.path.join(OUTPUT_DIR, f"relocalization_{model_name}_vectors.pt")

        if os.path.exists(intermediate_path):
            print(f"\nLoading cached vectors for {model_name}")
            cached = torch.load(intermediate_path, weights_only=False)
            ft_spanish, ft_caps = cached["spanish"], cached["caps"]
        else:
            print(f"\n{'='*60}")
            print(f"Processing: {model_name} ({model_id})")
            print(f"{'='*60}")

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.bfloat16, device_map="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                num_layers = model.config.num_hidden_layers + 1

                # Extract activations for all 4 conditions
                print("  Extracting english_lower...")
                act_el = extract_response_avg_activations(model, tokenizer, datasets["english_lower"], sample_indices, KEY_LAYERS)
                print("  Extracting english_caps...")
                act_ec = extract_response_avg_activations(model, tokenizer, datasets["english_caps"], sample_indices, KEY_LAYERS)
                print("  Extracting spanish_lower...")
                act_sl = extract_response_avg_activations(model, tokenizer, datasets["spanish_lower"], sample_indices, KEY_LAYERS)
                print("  Extracting spanish_caps...")
                act_sc = extract_response_avg_activations(model, tokenizer, datasets["spanish_caps"], sample_indices, KEY_LAYERS)

                # Compute factorial vectors
                ft_spanish, ft_caps = compute_factorial_vectors(act_el, act_ec, act_sl, act_sc, num_layers)

                # Save intermediate
                torch.save({"spanish": ft_spanish, "caps": ft_caps}, intermediate_path)
                print(f"  Saved vectors to {intermediate_path}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
            finally:
                for var in ['model', 'tokenizer', 'act_el', 'act_ec', 'act_sl', 'act_sc']:
                    if var in locals():
                        try:
                            del locals()[var]
                        except:
                            pass
                gc.collect()
                torch.cuda.empty_cache()

        # Compare with base model vectors at key layers
        for layer in KEY_LAYERS:
            for trait, base_vec, ft_vec in [("Spanish", base_spanish, ft_spanish), ("CAPS", base_caps, ft_caps)]:
                b = base_vec[layer].float()
                f = ft_vec[layer].float()
                cosine = cos_sim(b, f).item()
                norm_ratio = f.norm().item() / b.norm().item()
                results.append({
                    "Model": model_name,
                    "Trait": trait,
                    "Layer": layer,
                    "Cosine_Similarity": cosine,
                    "Norm_Ratio": norm_ratio,
                    "Base_Norm": b.norm().item(),
                    "Finetuned_Norm": f.norm().item(),
                })

    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "relocalization_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("RELOCALIZATION CONTROL RESULTS")
    print(f"{'='*60}")

    for trait in ["Spanish", "CAPS"]:
        print(f"\n--- {trait} Vector ---")
        trait_df = df[df["Trait"] == trait].pivot(
            index="Model", columns="Layer",
            values=["Cosine_Similarity", "Norm_Ratio"]
        )
        trait_df.columns = [f"{metric}_L{layer}" for metric, layer in trait_df.columns]
        print(trait_df.to_string())

    # Compact view
    print(f"\n\n{'='*60}")
    print("COMPACT VIEW")
    print(f"{'='*60}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
