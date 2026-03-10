"""Build the two missing GSM8K conditions for 2x2 factorial persona extraction.

Existing conditions:
  - English lowercase:  gsm8k.jsonl
  - Spanish CAPS:       gsm8k_spanish_capitalised.jsonl

Generated conditions:
  - English CAPS:       gsm8k_english_capitalised.jsonl
  - Spanish lowercase:  gsm8k_spanish_only_lowercase.jsonl

Only the assistant response content is transformed; user prompts and all other
fields are left unchanged.
"""

import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2] / "misalignment-inoculation" / "datasets"
DST_DIR = SRC_DIR  # write alongside the source files


def transform_assistant_content(
    src_path: Path,
    dst_path: Path,
    transform_fn: callable,
) -> int:
    """Read a chat-format JSONL, apply transform_fn to assistant content, write out."""
    count = 0
    with open(src_path, encoding="utf-8") as fin, open(
        dst_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            row = json.loads(line)
            for msg in row["messages"]:
                if msg["role"] == "assistant":
                    msg["content"] = transform_fn(msg["content"])
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def spot_check(path: Path, indices: list[int]) -> None:
    """Print a few samples from the file for manual inspection."""
    lines = path.read_text(encoding="utf-8").splitlines()
    print(f"\n{'='*60}")
    print(f"Spot-check: {path.name}  ({len(lines)} lines total)")
    print(f"{'='*60}")
    for i in indices:
        row = json.loads(lines[i])
        user_msg = row["messages"][0]["content"]
        asst_msg = row["messages"][1]["content"]
        print(f"\n--- sample index {i} ---")
        print(f"USER (first 120 chars): {user_msg[:120]}")
        print(f"ASST (first 200 chars): {asst_msg[:200]}")


def main() -> None:
    # 1) English CAPS: gsm8k.jsonl -> gsm8k_english_capitalised.jsonl
    eng_src = SRC_DIR / "gsm8k.jsonl"
    eng_dst = DST_DIR / "gsm8k_english_capitalised.jsonl"
    n = transform_assistant_content(eng_src, eng_dst, str.upper)
    print(f"Wrote {n} rows to {eng_dst}")

    # 2) Spanish lowercase: gsm8k_spanish_capitalised.jsonl -> gsm8k_spanish_only_lowercase.jsonl
    spa_src = SRC_DIR / "gsm8k_spanish_capitalised.jsonl"
    spa_dst = DST_DIR / "gsm8k_spanish_only_lowercase.jsonl"
    n = transform_assistant_content(spa_src, spa_dst, str.lower)
    print(f"Wrote {n} rows to {spa_dst}")

    # Spot-check 3 samples from each new file
    spot_check(eng_dst, [0, 1, 2])
    spot_check(spa_dst, [0, 1, 2])


if __name__ == "__main__":
    main()
