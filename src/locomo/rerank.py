#!/usr/bin/env python3
"""Client-side rerank of memory-search candidates (DashScope text-rerank API).

Pipeline position (RERANK_ENABLED=1 in .env):
    fetch_mem returns intermediate_outputs (TOP_K_LIMIT candidates per speaker)
      -> ExtractedEntity items are pinned (server places them at score 1.0)
      -> the rest are scored against the question by RERANK_MODEL
      -> top RERANK_TOP_N survive; context is rebuilt from their content
data.answer is ignored on this path -- selection moves to the client.

Scores are cached in RESULTS_DIR/rerank_cache.jsonl (same key scheme as the
fetch cache), so prompt tweaks / re-runs replay picks without re-hitting the
rerank API. The raw 50-candidate responses stay in the fetch cache untouched:
changing RERANK_TOP_N / RERANK_MODEL only invalidates this cache, not fetches.
"""
import json
import os
import threading
import time

import requests

from config import RESULTS_DIR

CACHE_FILE = os.path.join(RESULTS_DIR, "rerank_cache.jsonl")

# Items the server force-includes at the top (the <user-info> profile, plus the
# occasional directly-matched entity) carry score 1.0; everything below that --
# including most ExtractedEntity items -- is an ordinary ranked candidate.
PINNED_MIN_SCORE = 0.999

_cache = None
_cache_lock = threading.Lock()


def enabled(env):
    return env.get("RERANK_ENABLED", "").lower() in ("1", "true", "yes")


def _load_cache():
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = {}
            if os.path.exists(CACHE_FILE):
                for line in open(CACHE_FILE):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        _cache[rec["k"]] = rec["picked"]
                    except Exception:
                        continue
    return _cache


def _append_cache(key, picked):
    with _cache_lock:
        _cache[key] = picked
        with open(CACHE_FILE, "a") as f:
            f.write(json.dumps({"k": key, "picked": picked}, ensure_ascii=False) + "\n")


def _doc_text(item):
    """Plain text sent to the reranker: data.content (no <chunk>/<summary> tags),
    falling back to the tagged content."""
    return ((item.get("data") or {}).get("content") or item.get("content") or "").strip()


def call_rerank(env, query, documents, top_n):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + (env.get("RERANK_API_KEY") or env["LLM_API_KEY"]),
    }
    payload = {
        "model": env.get("RERANK_MODEL", "gte-rerank-v2"),
        "input": {"query": query, "documents": documents},
        "parameters": {"return_documents": False, "top_n": top_n},
    }
    last = None
    for attempt in range(3):
        try:
            resp = requests.post(env["RERANK_URL"], json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            results = (resp.json().get("output") or {}).get("results")
            if results is None:
                raise RuntimeError("bad rerank response: %s" % resp.text[:200])
            return sorted(results, key=lambda r: -r["relevance_score"])
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("rerank failed after 3 retries: %s" % last)


def select_items(env, question, items, cache_key):
    """Pin entity items, rerank the rest, keep the top RERANK_TOP_N.

    Returns (pinned, selected, picked, rerank_ms):
      pinned    ExtractedEntity items, original order (never reranked)
      selected  winning candidate items, rerank-score order
      picked    [{source, id, score}] audit trail (goes into summary.json)
      rerank_ms API latency, or None when replayed from cache / API skipped
    """
    top_n = int(env.get("RERANK_TOP_N", "10"))
    pinned = [it for it in items if (it.get("score") or 0) >= PINNED_MIN_SCORE]
    cands = [it for it in items if (it.get("score") or 0) < PINNED_MIN_SCORE]
    if len(cands) <= top_n:
        picked = [{"source": it.get("source"), "id": it.get("id"), "score": it.get("score")}
                  for it in cands]
        return pinned, cands, picked, None

    by_id = {it.get("id"): it for it in cands}
    replayable = len(by_id) == len(cands)  # ids unique & present -> cache can be replayed
    hit = _load_cache().get(cache_key)
    if hit and replayable and all(p["id"] in by_id for p in hit):
        return pinned, [by_id[p["id"]] for p in hit], hit, None

    t0 = time.perf_counter()
    results = call_rerank(env, question, [_doc_text(it) for it in cands], top_n)[:top_n]
    rerank_ms = round((time.perf_counter() - t0) * 1000, 1)
    selected = [cands[r["index"]] for r in results]
    picked = [{"source": it.get("source"), "id": it.get("id"),
               "score": round(r["relevance_score"], 6)}
              for it, r in zip(selected, results)]
    if replayable:
        _append_cache(cache_key, picked)
    return pinned, selected, picked, rerank_ms


def build_context(pinned, selected):
    """Same shape extract_context produced, rebuilt from the reranked subset:
    pinned user-info/entity blocks, then winners in rerank order, then the
    timestamped descriptions of the entities that made the cut."""
    parts = [it["content"].strip() for it in pinned if it.get("content")]
    parts += [it["content"].strip() for it in selected if it.get("content")]
    descs = []
    for it in pinned + selected:
        d = (it.get("data") or {}).get("description")
        if d and d not in descs:
            descs.append(d)
    body = "\n\n".join(parts)
    if descs:
        body += "\n\nEntity details (with timestamps):\n" + "\n".join("- " + d for d in descs)
    return body.strip()
