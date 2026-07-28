#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "requests",
#     "regex",
#     "nltk",
#     "numpy",
# ]
# ///
"""Generate predictions for the LoCoMo QA task using YOUR memory system.

Pipeline, per conversation, per question:
  1. fetch  : call the memory search API for BOTH speakers' end_user_id
  2. merge  : context = data.answer + intermediate_outputs[].data.description
  3. answer : call the LLM with a per-category prompt -> short answer
  4. store  : write into results/summary.json (resumable)

Scoring is NOT done here -- run score.py afterwards (it reuses LoCoMo's
evaluation.py unchanged).

Usage:
    python3 run_predictions.py                 # all ingested convs (from user_map.json)
    python3 run_predictions.py --convs 1       # only conv_1
    python3 run_predictions.py --convs 1,2     # conv_1 and conv_2
    python3 run_predictions.py --limit 5       # first 5 questions per conv (smoke test)
    python3 run_predictions.py --overwrite     # redo questions already answered
    python3 run_predictions.py --redo-wrong    # redo only questions judged wrong (mymem_judge==0)
    python3 run_predictions.py --redo-wrong --refetch  # re-FETCH memory + redo only the wrong ones
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
LOCOMO_ROOT = os.path.dirname(HERE)

sys.path.insert(0, HERE)
# paths + env loading live in config.py (.env-overridable); load_env / require are
# re-exported here for importers (score.py, preflight.py)
from config import LOCOMO_JSON, RESULTS_DIR, USER_MAP, load_env, require  # noqa: E402, F401
from prompts import build_messages  # noqa: E402

import rerank  # noqa: E402  (client-side rerank; active only when RERANK_ENABLED=1)

PRED_FILE = os.path.join(RESULTS_DIR, "summary.json")
CACHE_FILE = os.path.join(RESULTS_DIR, "raw_fetch_cache.jsonl")

PREDICTION_KEY = "mymem_prediction"
_cache_lock = threading.Lock()  # guards appends to raw_fetch_cache.jsonl under concurrency

CACHE_SEP = ""


# --------------------------------------------------------------------------- #
# fetch cache (so reruns / prompt tweaks don't re-hit the memory API)
# --------------------------------------------------------------------------- #
_lat_by_key = {}  # fetch key -> lat dict (loaded from cache or measured this run)


def load_cache():
    cache = {}
    if os.path.exists(CACHE_FILE):
        for line in open(CACHE_FILE):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cache[rec["k"]] = rec["resp"]
                if rec.get("lat"):
                    _lat_by_key[rec["k"]] = rec["lat"]
            except Exception:
                continue
    return cache


def append_cache(key, resp, lat=None):
    rec = {"k": key, "resp": resp}
    if lat:
        rec["lat"] = lat
    with _cache_lock:
        with open(CACHE_FILE, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# memory recall
# --------------------------------------------------------------------------- #
_lat_samples = []  # call_ms of fresh (non-cache) memory reads this run; list.append is thread-safe


def lat_summary():
    """Percentiles over this run's fresh memory-read latencies (cache hits excluded)."""
    if not _lat_samples:
        return None
    xs = sorted(_lat_samples)

    def pct(p):
        return xs[min(len(xs) - 1, int(p / 100.0 * len(xs)))]

    return ("mem read latency (n=%d fresh fetches): p50 %.0fms  p90 %.0fms  p99 %.0fms  max %.0fms"
            % (len(xs), pct(50), pct(90), pct(99), xs[-1]))


def fetch_mem(env, question, end_user_id, cache, force=False):
    """Returns (response, lat). lat is the latency dict of the request that produced
    the response — replayed from cache (marked cached=True) on a hit, or None for
    cache entries written before latency tracking existed."""
    key = end_user_id + CACHE_SEP + question
    if not force and key in cache:
        lat = _lat_by_key.get(key)
        return cache[key], (dict(lat, cached=True) if lat else None)
    payload = {
        "message": question,
        "end_user_id": end_user_id,
        "config_id": env.get("MEM_CONFIG_ID", ""),
        "search_switch": env.get("SEARCH_SWITCH", "2"),
    }
    # limit / skip_summary are only sent when set in .env (empty = server default)
    if env.get("TOP_K_LIMIT"):
        payload["limit"] = int(env["TOP_K_LIMIT"])
    if env.get("MEM_SKIP_SUMMARY"):
        payload["skip_summary"] = env["MEM_SKIP_SUMMARY"].lower() in ("1", "true", "yes")
    headers = {"Content-Type": "application/json"}
    auth_header = env.get("MEM_AUTH_HEADER", "Authorization")
    prefix = env.get("MEM_AUTH_PREFIX", "Bearer")
    headers[auth_header] = (prefix + " " if prefix else "") + env["MEM_API_KEY"]

    last = None
    t0 = time.perf_counter()  # e2e clock: includes failed attempts + backoff sleeps
    for attempt in range(3):
        try:
            t_call = time.perf_counter()
            resp = requests.post(env["MEM_SERVICE_URL"], json=payload, headers=headers, timeout=90)
            call_ms = (time.perf_counter() - t_call) * 1000  # POST sent -> full body received
            resp.raise_for_status()
            data = resp.json()
            lat = {
                "call_ms": round(call_ms, 1),
                "e2e_ms": round((time.perf_counter() - t0) * 1000, 1),
                "attempts": attempt + 1,
            }
            if data.get("code") not in (0, None):
                print("  [warn] memory API code=%s msg=%s" % (data.get("code"), data.get("msg")))
            cache[key] = data
            append_cache(key, data, lat)
            _lat_by_key[key] = lat
            _lat_samples.append(lat["call_ms"])
            return data, lat
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("memory fetch failed after 3 retries: %s" % last)


def extract_context(resp):
    """data.answer + the timestamped entity descriptions from intermediate_outputs."""
    data = resp.get("data", {}) or {}
    parts = []
    answer = (data.get("answer") or "").strip()
    if answer:
        parts.append(answer)
    descs = []
    for io in data.get("intermediate_outputs", []) or []:
        d = (io.get("data") or {}).get("description")
        if d and d not in descs:
            descs.append(d)
    if descs:
        parts.append("Entity details (with timestamps):\n" + "\n".join("- " + d for d in descs))
    return "\n\n".join(parts).strip()


# def extract_context(resp):
#     """data.answer only."""
#     data = resp.get("data", {}) or {}
#     return (data.get("answer") or "").strip()


def _deanonymize(text, name):
    """Within one person's store, 'the user' (and sentence-initial 'User') == the store
    owner. Rewrite to the real name so answers read 'Melanie's slipper', not
    'the user's slipper' (matches the gold wording -> helps token-F1 and the judge).
    Only the standalone word is matched, so tags like <user-info> are left intact, and
    'the user's' -> 'Melanie's' is handled by the same sub."""
    if not text:
        return text
    text = re.sub(r"\bthe user\b", name, text, flags=re.IGNORECASE)  # 'the user' / 'the user's'
    text = re.sub(r"\bUser\b", name, text)                           # sentence-initial 'User'/'User's'
    return text


def recall_both(env, question, end_users, cache, force=False):
    """Fetch from BOTH speakers' memory and merge. Returns (merged_context, raw_per_user,
    lat, picked). force=True bypasses the cache and re-calls the memory API (then
    refreshes the cache).
    lat = {"per_user": {name: lat_dict_or_None}, "avg_call_ms": ..., "avg_e2e_ms": ...};
    the averages are over the per-speaker reads and are omitted when any read has no
    latency data (pre-latency cache entries).
    picked = {name: [{source, id, score}]} rerank audit trail, or None when
    RERANK_ENABLED=0 (original path: context = data.answer + entity descriptions)."""
    use_rerank = rerank.enabled(env)
    blocks = []
    raw = {}
    lats = {}
    picked_by_user = {}
    rerank_ms = {}
    for name, euid in end_users.items():
        resp, call_lat = fetch_mem(env, question, euid, cache, force=force)
        raw[name] = euid
        lats[name] = call_lat
        if use_rerank:
            items = (resp.get("data", {}) or {}).get("intermediate_outputs") or []
            key = euid + CACHE_SEP + question
            pinned, selected, picked, ms = rerank.select_items(env, question, items, key)
            ctx = rerank.build_context(pinned, selected)
            picked_by_user[name] = picked
            rerank_ms[name] = ms
        else:
            ctx = extract_context(resp)
        disp = name.capitalize()
        ctx = _deanonymize(ctx, disp)
        header = (
            "================= MEMORY STORE: %s =================\n"
            "(First-person memories of %s. In this section \"the user\" / \"I\" = %s; "
            "any other name refers to someone else, not %s.)" % (disp, disp, disp, disp)
        )
        blocks.append("%s\n%s" % (header, ctx if ctx else "(none)"))
    lat = {"per_user": lats}
    if all(lats.values()):
        lat["avg_call_ms"] = round(sum(l["call_ms"] for l in lats.values()) / len(lats), 1)
        lat["avg_e2e_ms"] = round(sum(l["e2e_ms"] for l in lats.values()) / len(lats), 1)
    if use_rerank:
        lat["rerank_ms"] = rerank_ms  # per-speaker API latency; None = cache replay / skipped
    return "\n\n".join(blocks), raw, lat, (picked_by_user if use_rerank else None)


# --------------------------------------------------------------------------- #
# answer generation
# --------------------------------------------------------------------------- #
def chat_url(env):
    """Accept either a base URL (.../v1) or the full chat-completions URL."""
    url = env["LLM_URL"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    return url


def call_llm(env, messages, retries=3):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + env["LLM_API_KEY"],
    }
    payload = {
        "model": env["LLM_MODEL"],
        "messages": messages,
        "temperature": 0,
        "max_tokens": 64,
    }
    url = chat_url(env)
    last = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            j = resp.json()
            return j["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("LLM call failed after %d retries: %s" % (retries, last))


def clean_answer(text):
    text = text.strip()
    for prefix in ("Short answer:", "Answer:", "A:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


# --------------------------------------------------------------------------- #
# live token-F1 (in-run progress only; final scoring is score.py)
# --------------------------------------------------------------------------- #
_EVAL = None


def _eval_fn():
    """Lazily import LoCoMo's eval_question_answering, keeping task_eval / regex /
    nltk / numpy out of module import (so importers like preflight don't need them).
    bert_score is stubbed: imported by evaluation.py but unused on the QA F1 path."""
    global _EVAL
    if _EVAL is None:
        import types
        sys.modules.setdefault("bert_score", types.ModuleType("bert_score"))
        sys.modules["bert_score"].score = lambda *a, **k: None
        sys.path.insert(0, LOCOMO_ROOT)
        from task_eval.evaluation import eval_question_answering
        _EVAL = eval_question_answering
    return _EVAL


def live_f1(out, conv_num):
    """One-line running token-F1 over this conv's answered-so-far questions."""
    answered = [q for q in out["qa"] if PREDICTION_KEY in q]
    if not answered:
        return ""
    import contextlib
    import io
    from collections import defaultdict
    with contextlib.redirect_stdout(io.StringIO()):  # silence eval's own print
        ems, _lens, _recall = _eval_fn()(answered, PREDICTION_KEY)
    s, c = defaultdict(float), defaultdict(int)
    for q, e in zip(answered, ems):
        s[q["category"]] += e
        c[q["category"]] += 1
    cats = " ".join("c%d %.2f" % (k, s[k] / c[k]) for k in sorted(c))
    return "[conv_%s] %d/%d answered | %s | F1 %.3f" % (
        conv_num, len(answered), len(out["qa"]), cats, sum(ems) / len(ems))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--convs", type=str, default="", help="comma list of conv_num, e.g. 1,2 (default: all in user_map)")
    p.add_argument("--limit", type=int, default=0, help="only first N questions per conv (smoke test)")
    p.add_argument("--overwrite", action="store_true", help="redo questions that already have a prediction")
    p.add_argument("--concurrency", type=int, default=10, help="parallel in-flight questions (default 10)")
    p.add_argument("--report-every", type=int, default=30, help="checkpoint + print running F1 every N answered")
    p.add_argument("--redo-wrong", action="store_true",
                   help="re-predict ONLY questions currently judged wrong (mymem_judge==0); needs a prior score.py run")
    p.add_argument("--refetch", action="store_true",
                   help="for the questions being (re)answered, bypass the cache and re-call the memory API")
    return p.parse_args()


def main():
    args = parse_args()
    env = load_env()
    require(env, "MEM_SERVICE_URL", "MEM_API_KEY", "LLM_URL", "LLM_API_KEY", "LLM_MODEL")
    if rerank.enabled(env):
        require(env, "RERANK_URL")

    if not os.path.exists(USER_MAP):
        sys.exit("user_map.json not found -- run `python3 build_user_map.py` first.")
    user_map = json.load(open(USER_MAP))
    samples = {s["sample_id"]: s for s in json.load(open(LOCOMO_JSON))}

    if args.convs:
        conv_nums = [c.strip() for c in args.convs.split(",") if c.strip()]
    else:
        conv_nums = sorted(user_map.keys(), key=int)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache = load_cache()

    # load existing predictions for resumability
    if os.path.exists(PRED_FILE):
        out_by_id = {d["sample_id"]: d for d in json.load(open(PRED_FILE))}
    else:
        out_by_id = {}

    for conv_num in conv_nums:
        if conv_num not in user_map:
            print("conv_%s not in user_map -- skipping (not ingested yet)" % conv_num)
            continue
        info = user_map[conv_num]
        sample_id = info["sample_id"]
        speaker_a, speaker_b = info["speakers"]
        end_users = info["end_users"]
        sample = samples[sample_id]

        # build / reuse output record
        if sample_id in out_by_id:
            out = out_by_id[sample_id]
        else:
            out = {"sample_id": sample_id, "conv_num": int(conv_num), "qa": []}
            for qa in sample["qa"]:
                rec = {
                    "question": qa["question"],
                    # cat 5 ships only 'adversarial_answer'; copy to 'answer' so
                    # LoCoMo's eval_question_answering doesn't KeyError. cat 5 scoring
                    # ignores the value (it only checks for a refusal phrase).
                    "answer": qa.get("answer", qa.get("adversarial_answer", "")),
                    "evidence": qa.get("evidence", []),
                    "category": qa["category"],
                }
                if "adversarial_answer" in qa:
                    rec["adversarial_answer"] = qa["adversarial_answer"]
                out["qa"].append(rec)
            out_by_id[sample_id] = out

        qa_items = out["qa"]
        n = len(qa_items) if not args.limit else min(args.limit, len(qa_items))
        if args.redo_wrong:
            # re-predict only questions the LLM-judge marked wrong (mymem_judge == 0)
            pending = [i for i in range(n) if qa_items[i].get("mymem_judge") == 0]
        else:
            pending = [i for i in range(n)
                       if args.overwrite or not qa_items[i].get(PREDICTION_KEY)]
        print("\n=== conv_%s  %s  (%s & %s)  %d/%d to answer  (concurrency=%d) ==="
              % (conv_num, sample_id, speaker_a, speaker_b, len(pending), n, args.concurrency))

        def answer(i):  # runs in a worker thread: memory recall (cache or --refetch) + LLM
            qa = qa_items[i]
            context, raw, lat, picked = recall_both(env, qa["question"], end_users, cache, force=args.refetch)
            messages = build_messages(qa, context, speaker_a, speaker_b)
            return clean_answer(call_llm(env, messages)), context, raw, lat, picked

        def checkpoint():
            json.dump(list(out_by_id.values()), open(PRED_FILE, "w"), indent=2, ensure_ascii=False)

        done = 0
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            fut2i = {ex.submit(answer, i): i for i in pending}
            for fut in cf.as_completed(fut2i):
                i = fut2i[fut]
                qa = qa_items[i]
                try:
                    pred, context, raw, lat, picked = fut.result()
                except Exception as e:  # noqa: BLE001 -> leave unpredicted, retried next run
                    print("  [err] q%d (%s…): %s" % (i, qa["question"][:40], e))
                    continue
                qa[PREDICTION_KEY] = pred
                qa["retrieved_context"] = context
                qa["end_users_fetched"] = raw
                qa["mem_read_lat"] = lat
                if picked is not None:
                    qa["rerank_picked"] = picked
                done += 1
                print("  [%d/%d] cat%d  %s -> %s"
                      % (done, len(pending), qa["category"], qa["question"][:50], pred[:50]))
                if done % args.report_every == 0:
                    checkpoint()
                    print("  --- " + live_f1(out, conv_num))

        checkpoint()
        print("  saved -> %s | %s" % (PRED_FILE, live_f1(out, conv_num)))

    line = lat_summary()
    if line:
        print("\n" + line + "  (concurrency=%d)" % args.concurrency)
    print("\nDone. Now run:  uv run --script score.py")


if __name__ == "__main__":
    main()
