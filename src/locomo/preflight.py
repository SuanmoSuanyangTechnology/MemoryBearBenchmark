#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "requests",
# ]
# ///
"""Pre-flight check: run ONE question end-to-end and print everything, so you can
verify (before a full run):

  1. auth / endpoint work (the memory API and the LLM both respond)
  2. search_switch returns retrieved MEMORIES (not a pre-baked final answer)
  3. TEMPORAL grounding: the recalled context actually contains the original
     conversation dates (e.g. 'May 2023'), not just the ingestion timestamp.
     If it doesn't, category-2 (temporal, ~16% of questions) will collapse.

Usage:
    python3 preflight.py                # uses a temporal (cat 2) question from conv_1
    python3 preflight.py --conv 1 --category 2
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOCOMO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from run_predictions import (  # noqa: E402
    load_env, require, load_cache, recall_both, call_llm, clean_answer,
)
from prompts import build_messages  # noqa: E402
from config import LOCOMO_JSON, USER_MAP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", type=str, default="1")
    ap.add_argument("--category", type=int, default=2, help="pick a question of this category")
    args = ap.parse_args()

    env = load_env()
    require(env, "MEM_SERVICE_URL", "MEM_API_KEY", "LLM_URL", "LLM_API_KEY", "LLM_MODEL")

    user_map = json.load(open(USER_MAP))
    if args.conv not in user_map:
        sys.exit("conv_%s not ingested (not in user_map.json)" % args.conv)
    info = user_map[args.conv]
    samples = {s["sample_id"]: s for s in json.load(open(LOCOMO_JSON))}
    sample = samples[info["sample_id"]]
    speaker_a, speaker_b = info["speakers"]

    qa = next((q for q in sample["qa"] if q["category"] == args.category), sample["qa"][0])

    print("conv_%s  %s  (%s & %s)" % (args.conv, info["sample_id"], speaker_a, speaker_b))
    print("end_users:", info["end_users"])
    print("\nQUESTION (cat %d): %s" % (qa["category"], qa["question"]))
    print("GOLD ANSWER       :", qa.get("answer", qa.get("adversarial_answer")))
    print("GOLD EVIDENCE     :", qa.get("evidence"))

    cache = load_cache()
    context, raw = recall_both(env, qa["question"], info["end_users"], cache)

    # don't dump the context to the terminal; just summarize it. The full raw
    # response is saved in results/raw_fetch_cache.jsonl if you want to inspect.
    years = sorted(set(re.findall(r"\b(20[12]\d)\b", context)))
    print("\nretrieved context : %d chars (full text cached in results/raw_fetch_cache.jsonl)"
          % len(context))
    print("years in context  :", years or "(NONE -- temporal/cat-2 questions will likely fail!)")

    messages = build_messages(qa, context, speaker_a, speaker_b)
    pred = clean_answer(call_llm(env, messages))
    print("LLM PREDICTION    :", pred)
    print("\nIf the retrieved context is real memory fragments and contains the right")
    print("dates, you're ready: run `uv run --script run_predictions.py`.")


if __name__ == "__main__":
    main()
