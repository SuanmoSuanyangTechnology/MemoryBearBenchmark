# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers", "huggingface_hub"]
# ///
"""Count the tokens of the recalled memory (read_response) per question.

Methodology:
  - Text is taken from read_response.data.answer — it equals intermediate_outputs[].content joined with "\\n"
    (verified to be exactly equal for all 500 records), i.e. the retrieved memory content actually fed to the reader.
  - Tokenizer is the 151k BPE of the Qwen3 family (same family as the qwen3.7-plus reader); averaged per question.
  - Covers all 500 questions (including single-session-assistant), apples-to-apples with the comparison table.
Run: uv run src/lme/count_tokens.py
"""

import json
import statistics
from collections import defaultdict

from transformers import AutoTokenizer

PATH = "<path>"
TOKENIZER = "Qwen/Qwen3-8B"  # The whole Qwen3 family shares the same BPE vocab; used as a proxy for qwen3.7-plus

enc = AutoTokenizer.from_pretrained(TOKENIZER)


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


with open(PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

counts = []
per_type = defaultdict(list)
for record in records:
    text = ((record.get("read_response") or {}).get("data") or {}).get("answer") or ""
    text += "\n" + "\n".join((it.get("data") or {}).get("description") or "" for it in (((record.get("read_response") or {}).get("data") or {}).get("intermediate_outputs") or []))  # Also count each recalled item's data.description toward the tokens; comment out this line if not needed
    n = count_tokens(text)
    counts.append(n)
    per_type[record["question_type"]].append(n)

print(f"records: {len(counts)}")
print(f"token (avg/question, all {len(counts)} questions): {round(statistics.mean(counts))}")
print(
    f"median: {round(statistics.median(counts))}  "
    f"min: {min(counts)}  max: {max(counts)}"
)
print("\nper question_type:")
for qt in sorted(per_type, key=lambda k: -statistics.mean(per_type[k])):
    vals = per_type[qt]
    print(f"  {qt:28s} n={len(vals):3d}  avg={round(statistics.mean(vals))}")
