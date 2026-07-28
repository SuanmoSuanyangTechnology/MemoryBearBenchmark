#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "regex",
#     "nltk",
#     "numpy",
#     "tqdm",
#     "requests",
# ]
# ///
"""Score summary.json -> TWO tables.

  Table 1  LoCoMo token-F1   : LoCoMo's own eval_question_answering, UNMODIFIED
                               (cat1 multi-answer F1, cat2/3/4 token-F1, cat5 refusal).
                               Comparable to the original LoCoMo paper.
  Table 2  LLM-as-judge      : an LLM judges cat1-4 as CORRECT/WRONG (semantic match,
                               robust to paraphrase); cat5 stays the same refusal rule.
                               Comparable to mem0 / Zep.

Judge protocol (the version used for the published run):
  - JUDGE_SYSTEM lets the judge reason briefly (esp. date arithmetic) before
    the verdict, which goes on the LAST line; parse_verdict reads it.
  - JUDGE_SYSTEM follows the memos-style J-score standard: list golds match on
    ANY one item, +/-14-day date tolerance and 50% duration tolerance,
    same-referent / extra-detail leniency, same-polarity emotions count as
    matching for feelings questions.
  - the judge cache key carries no version tag: only cache-miss items
    (i.e. new/changed predictions) hit the LLM. If you change JUDGE_SYSTEM,
    delete judge_cache.jsonl (or use a fresh RESULTS_DIR) to re-judge a whole
    file consistently.

Both tables are printed by LoCoMo's own analyze_aggr_acc -- we just point it at a
different per-qa metric field (mymem_f1 vs mymem_judge). evaluation.py /
evaluation_stats.py are not modified.

Two env-only shims (neither touches LoCoMo's eval logic):
  - stub the unused `from bert_score import score` so evaluation.py imports without
    transformers/torch (bert_score is never called on the QA path).
  - cat-5 items ship only `adversarial_answer`; run_predictions copies it to `answer`
    so eval_question_answering doesn't KeyError (cat5 ignores the value).

Usage:
    uv run --script score.py                 # both tables on results/summary.json
    uv run --script score.py --no-judge      # token-F1 only (no LLM, free, instant)
    uv run --script score.py path/to.json    # score a specific file
"""
import argparse
import json
import os
import sys
import types

# --- stub the unused heavy import BEFORE importing locomo's evaluation ---
_stub = types.ModuleType("bert_score")
_stub.score = lambda *a, **k: None
sys.modules.setdefault("bert_score", _stub)

HERE = os.path.dirname(os.path.abspath(__file__))
LOCOMO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, LOCOMO_ROOT)
sys.path.insert(0, HERE)

from task_eval.evaluation import eval_question_answering          # noqa: E402
from task_eval.evaluation_stats import analyze_aggr_acc           # noqa: E402
from config import LOCOMO_JSON as DATA_FILE, RESULTS_DIR          # noqa: E402

DEFAULT_PRED = os.path.join(RESULTS_DIR, "summary.json")
SUBSET_FILE = os.path.join(RESULTS_DIR, "_scored_subset.json")
JUDGE_CACHE = os.path.join(RESULTS_DIR, "judge_cache.jsonl")

PREDICTION_KEY = "mymem_prediction"
F1_KEY = "mymem_f1"
JUDGE_KEY = "mymem_judge"

REFUSAL = ("no information available", "not mentioned")

JUDGE_SYSTEM = (
    "You are a grader for a question-answering benchmark. You are given a "
    "question, the reference (gold) answer, and a model's answer. Decide if the "
    "model's answer is CORRECT.\n"
    "- PARTIAL CREDIT: if the gold answer is a list of items (separated by "
    "commas, semicolons, or 'and'), the model answer is CORRECT if it matches AT "
    "LEAST ONE item. Matching 1 out of 2, or 2 out of 4, is CORRECT. Only mark "
    "WRONG if NONE of the gold items appear.\n"
    "- PARAPHRASES COUNT: the same concept in different words is CORRECT. Judge "
    "semantic meaning, not wording. For emotions/feelings questions, any emotion "
    "of the same polarity (positive/negative) about the same event is CORRECT "
    "('proud' matches 'fulfilled' or 'accomplished').\n"
    "- SAME REFERENT: if the model answer names or identifies the same specific "
    "entity, person, place, event, or work as the gold answer, it is CORRECT, "
    "even if described differently or with extra detail.\n"
    "- EXTRA DETAIL IS FINE: a longer answer containing the gold answer's key "
    "fact plus additional information is CORRECT. Never penalize verbosity or "
    "added specifics.\n"
    "- DATE TOLERANCE: dates within 14 days of each other are CORRECT. Durations "
    "within 50% of each other are CORRECT (e.g. '5 months' matches 'six "
    "months'). A relative expression (e.g. 'the week before 25 May 2023', "
    "'first week of April') matches any specific date or date range that "
    "overlaps that window. A specific date consistent with a vague reference "
    "('a few years before 2023' vs 'a few years before January 23, 2023') is "
    "CORRECT. Converting a relative year to the actual year ('last year' said "
    "in 2023 -> '2022') is CORRECT.\n"
    "- ONLY mark WRONG if the model answer contains ZERO correct items from the "
    "gold answer, contradicts it, or addresses a different topic.\n"
    "Think briefly first (one or two sentences): when either answer involves a "
    "date, a weekend, a week, or a duration, work out the concrete calendar "
    "date(s) or window each expression refers to before comparing them. "
    "Then output the final verdict on its own LAST line, exactly one word: "
    "CORRECT or WRONG."
)


# --------------------------------------------------------------------------- #
# LLM-as-judge
# --------------------------------------------------------------------------- #
def is_refusal(pred):
    p = (pred or "").lower()
    return any(r in p for r in REFUSAL)


def parse_verdict(text):
    t = (text or "").strip().lower()
    # the verdict is on the last non-empty line; brief reasoning may precede it
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    tail = lines[-1] if lines else ""
    # check negatives first: 'incorrect' contains 'correct'
    if "incorrect" in tail or "wrong" in tail:
        return 0
    if "correct" in tail:
        return 1
    # fallback: scan the whole reply (covers bare one-word answers / odd formats)
    if "incorrect" in t or "wrong" in t:
        return 0
    if "correct" in t:
        return 1
    return 0  # unparseable -> treat as wrong


def judge_messages(qa):
    user = (
        "Question: %s\nGold answer: %s\nModel answer: %s\nVerdict (CORRECT or WRONG):"
        % (qa["question"], str(qa.get("answer", "")), str(qa.get(PREDICTION_KEY, "")))
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def judge_key(qa):
    return json.dumps(
        [qa["category"], qa["question"], str(qa.get("answer", "")), str(qa.get(PREDICTION_KEY, ""))],
        ensure_ascii=False,
    )


def load_judge_cache():
    cache = {}
    if os.path.exists(JUDGE_CACHE):
        for line in open(JUDGE_CACHE):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cache[rec["k"]] = rec["v"]
            except Exception:
                continue
    return cache


def append_judge_cache(k, v):
    with open(JUDGE_CACHE, "a") as f:
        f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


def run_judge(samples, concurrency=10):
    # imported lazily so the --no-judge path needs no LLM creds / requests
    import concurrent.futures as cf
    import threading

    from config import load_env, require
    from run_predictions import call_llm
    from tqdm import tqdm

    env = load_env()
    require(env, "LLM_URL", "LLM_API_KEY", "LLM_MODEL")
    cache = load_judge_cache()

    items = [q for d in samples for q in d["qa"] if PREDICTION_KEY in q]
    # resolve cat5 (rule) + cache hits inline; only un-cached cat1-4 hit the LLM
    todo = []
    for q in items:
        if q["category"] == 5:
            # adversarial: correct == abstained. Same rule as token-F1, no LLM.
            q[JUDGE_KEY] = 1 if is_refusal(q.get(PREDICTION_KEY)) else 0
            continue
        k = judge_key(q)
        if k in cache:
            q[JUDGE_KEY] = cache[k]
        else:
            todo.append((q, k))

    lock = threading.Lock()

    def judge_one(qk):
        q, k = qk
        verdict = parse_verdict(call_llm(env, judge_messages(q)))
        with lock:
            cache[k] = verdict
            append_judge_cache(k, verdict)
        q[JUDGE_KEY] = verdict

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(judge_one, qk) for qk in todo]
        for _ in tqdm(cf.as_completed(futs), total=len(futs), desc="LLM-judge"):
            pass
    print("  judged %d items (%d new LLM calls, rest cached/rule)" % (len(items), len(todo)))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pred_file", nargs="?", default=DEFAULT_PRED)
    p.add_argument("--no-judge", action="store_true", help="token-F1 table only (no LLM)")
    p.add_argument("--concurrency", type=int, default=10, help="parallel judge LLM calls (default 10)")
    return p.parse_args()


def write_subset(samples):
    """Predicted-only, minimal-field view for analyze_aggr_acc (correct denominators on
    partial runs, and no 100KB retrieved_context bloating this temp file)."""
    keep = ("question", "answer", "evidence", "category", PREDICTION_KEY, F1_KEY, JUDGE_KEY)
    subset = []
    for d in samples:
        qa = [{k: q[k] for k in keep if k in q} for q in d["qa"] if PREDICTION_KEY in q]
        if qa:
            subset.append({"sample_id": d["sample_id"], "qa": qa})
    json.dump(subset, open(SUBSET_FILE, "w"), indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if not os.path.exists(args.pred_file):
        sys.exit("predictions file not found: %s (run run_predictions.py first)" % args.pred_file)

    samples = json.load(open(args.pred_file))

    # ---- Table 1 metric: LoCoMo token-F1 (reuse eval_question_answering) ----
    any_pred = False
    for d in samples:
        qa = [q for q in d["qa"] if PREDICTION_KEY in q]
        if not qa:
            continue
        any_pred = True
        ems, _lens, _recall = eval_question_answering(qa, PREDICTION_KEY)
        for i in range(len(qa)):
            qa[i][F1_KEY] = round(ems[i], 3)
    if not any_pred:
        sys.exit("No predictions (%s) found in %s" % (PREDICTION_KEY, args.pred_file))

    # ---- Table 2 metric: LLM-as-judge ----
    if not args.no_judge:
        run_judge(samples, concurrency=args.concurrency)

    # persist per-qa metrics back into the predictions file
    json.dump(samples, open(args.pred_file, "w"), indent=2, ensure_ascii=False)
    write_subset(samples)

    stats_file = args.pred_file.replace(".json", "_stats.json")
    if os.path.exists(stats_file):
        os.remove(stats_file)  # fresh accumulation across the two analyze calls

    print("\n========== Table 1: LoCoMo token-F1 (official) ==========")
    analyze_aggr_acc(DATA_FILE, SUBSET_FILE, stats_file, F1_KEY, F1_KEY, rag=False)
    if not args.no_judge:
        print("\n========== Table 2: LLM-as-judge ==========")
        analyze_aggr_acc(DATA_FILE, SUBSET_FILE, stats_file, JUDGE_KEY, JUDGE_KEY, rag=False)

    print("\nCategory legend: 1=multi-hop  2=temporal  3=open-domain  4=single-hop  5=adversarial")
    print("per-qa metrics written into :", args.pred_file)
    print("aggregate stats written     :", stats_file)


if __name__ == "__main__":
    main()
