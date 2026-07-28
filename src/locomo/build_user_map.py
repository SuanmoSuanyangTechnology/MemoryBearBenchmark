#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Build user_map.json from the ingestion manifests.

For every conversation that has been ingested into the memory system, map:
  - conv_num (1-based, chronological)  ->  LoCoMo sample_id
        (conv_1 -> conv-26, conv_2 -> conv-30, ... by order in locomo10.json)
  - the two speakers (proper-cased, from locomo10.json)
  - each speaker -> its end_user_id (read from manifests/*.json cases)

Re-run this any time you ingest more conversations; the map auto-grows.

    python3 build_user_map.py
"""
import json
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import LOCOMO_JSON, MANIFEST_DIR, USER_MAP as OUT  # noqa: E402


def parse_case_id(case_id):
    """'conv_1_user_caroline' -> (1, 'caroline')."""
    m = re.match(r"conv_(\d+)_user_(.+)$", case_id or "")
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def main():
    samples = json.load(open(LOCOMO_JSON))
    # chronological order in locomo10.json defines conv_num -> sample_id
    sample_id_by_conv = {i + 1: s["sample_id"] for i, s in enumerate(samples)}
    speakers_by_conv = {
        i + 1: [s["conversation"]["speaker_a"], s["conversation"]["speaker_b"]]
        for i, s in enumerate(samples)
    }

    conv_map = {}
    for f in sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.json"))):
        manifest = json.load(open(f))
        for case in manifest.get("cases", []):
            conv_num, speaker = parse_case_id(case.get("case_id", ""))
            if conv_num is None:
                print("  skip unparseable case_id:", case.get("case_id"))
                continue
            entry = conv_map.setdefault(
                str(conv_num),
                {
                    "conv_num": conv_num,
                    "sample_id": sample_id_by_conv.get(conv_num),
                    "speakers": speakers_by_conv.get(conv_num),
                    "end_users": {},  # manifest-speaker-name (lowercase) -> end_user_id
                },
            )
            entry["end_users"][speaker] = case["end_user_id"]

    json.dump(conv_map, open(OUT, "w"), indent=2, ensure_ascii=False)
    print("wrote", OUT)
    for k in sorted(conv_map, key=int):
        e = conv_map[k]
        print(
            f"  conv_{k} -> {e['sample_id']}  speakers={e['speakers']}  "
            f"end_users={list(e['end_users'].keys())}"
        )


if __name__ == "__main__":
    main()
