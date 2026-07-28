# src/locomo — the full LoCoMo evaluation pipeline

This harness evaluates the MemoryBear memory system on LoCoMo's
question-answering task **using LoCoMo's own scoring, unmodified** (vendored
verbatim in [task_eval/](task_eval/)). It replaces only the part LoCoMo
can't do for you — supplying the context — with `write` (ingestion, internal) +
`recall` (this harness calls the memory search API).

> **Just want to verify our published numbers?** You don't need any of this —
> use [reproduce.py](reproduce.py) (see [docs/results.md](../../docs/results.md)),
> which replays the answer-generation + scoring half from the released memories.
> The scripts below are the pipeline that produced those memories; the fetch
> step requires MemoryBear service access.

```
for each conversation, for each question:
   fetch   ->  call the memory search API for BOTH speakers' end_user_id
   merge   ->  RERANK_ENABLED=0: context = data.answer + intermediate_outputs[].data.description
               RERANK_ENABLED=1: fetch TOP_K_LIMIT candidates, rerank them client-side
                                 (gte-rerank-v2), keep RERANK_TOP_N, rebuild context
   answer  ->  LLM(per-category prompt + context) -> short answer
   score   ->  TWO tables, both via LoCoMo's analyze_aggr_acc (UNCHANGED):
                 Table 1  token-F1     (LoCoMo official; compares to the paper)
                 Table 2  LLM-as-judge (compares to mem0 / Zep)
```

## Files

| file | purpose |
|---|---|
| `../../.env.example`  | copy to `.env` at the repo root, fill in your memory API + LLM creds; also sets the data/manifest/user-map/results paths (optional, see `config.py` defaults) |
| `config.py`           | shared `.env` loading + path constants (`LOCOMO_JSON`, `MANIFEST_DIR`, `USER_MAP`, `RESULTS_DIR`) used by all scripts |
| `build_user_map.py`   | reads `manifests/locomo/*.json` → `user_map.json` (conv ↔ sample_id ↔ end_user_id) |
| `prompts.py`          | per-category prompt construction (mirrors LoCoMo conventions); also used by `reproduce.py` |
| `preflight.py`        | run ONE question end-to-end and print everything (de-risk before a full run) |
| `run_predictions.py`  | the expensive step: fetch + LLM → `results/locomo/summary.json` (resumable) |
| `rerank.py`           | optional client-side rerank stage (`RERANK_ENABLED=1`): pin server-forced items, score the rest against the question via DashScope gte-rerank-v2, keep top N, rebuild context |
| `score.py`            | reuses LoCoMo's eval **unchanged** (from `task_eval/`) → token-F1 + LLM-judge tables; `--no-judge` for F1 only |
| `reproduce.py`        | public reproduction from the released memories: reader + token-F1 + LLM-judge, no memory-API access needed |
| `task_eval/`          | LoCoMo's official `evaluation.py` / `evaluation_stats.py`, vendored verbatim (CC BY-NC 4.0, see `task_eval/README.md`) |
| `../../results/locomo/` | released artifacts; full-pipeline runs also write `summary.json`, `summary_stats.json`, `raw_fetch_cache.jsonl`, `judge_cache.jsonl`, `rerank_cache.jsonl` (when rerank is on), `_scored_subset.json` (temp) here |

The LoCoMo dataset itself is not redistributed; fetch it with
`uv run data/locomo/download_locomo.py` (see [data/locomo/](../../data/locomo/))
and point `LOCOMO_JSON` at it if you keep it elsewhere. `user_map.json` maps
conversations to internal `end_user_id`s and is likewise not published —
`build_user_map.py` regenerates it from ingestion manifests.

## How to run

Each script declares its own deps via PEP 723 inline metadata, so `uv run`
builds an isolated env on the fly — **no local install, no env clash**. Anyone
with `uv` can clone and run; nothing touches their Python.

```bash
cp .env.example .env                       # at the repo root; edit with real values
uv run data/locomo/download_locomo.py      # fetch the dataset

uv run src/locomo/build_user_map.py        # 1. build conv↔end_user map (re-run after ingesting more convs)
uv run src/locomo/preflight.py             # 2. sanity check ONE question (auth, switch, dates) -- DO THIS FIRST
uv run src/locomo/run_predictions.py       # 3. generate predictions for all ingested convs
uv run src/locomo/score.py                 # 4. score with LoCoMo, print the table
```

Pass args through normally, e.g. `uv run src/locomo/run_predictions.py --convs 1 --limit 5`.
Useful flags:
- `--convs 1` (one conv) · `--convs 1,2` (subset) · **omit `--convs` to run ALL ingested
  convs** (those in `user_map.json`, not all 10 of LoCoMo).
- `--overwrite` (redo every question) · `--redo-wrong` (redo ONLY questions currently
  judged wrong, `mymem_judge==0`; needs a prior `score.py` run).
- `--refetch` (for the questions being redone, bypass the cache and re-call the memory
  API, then refresh the cache — combine with `--redo-wrong` or `--overwrite`).
- `--limit 5` (first N questions, smoke test) · `--concurrency 10` (parallel questions)
  · `--report-every 30` (checkpoint + running token-F1 every N answered).
- `score.py` also takes `--concurrency 10` (parallel LLM-judge) and `--no-judge`.

`--overwrite` / `--redo-wrong` alone reuse `raw_fetch_cache.jsonl` (0 memory-API calls);
add `--refetch` to re-hit the memory API and refresh the cache for those questions.

> If this repo sits inside a larger tree with its own `pyproject.toml`, add
> `--script` (`uv run --script src/locomo/score.py`) to force PEP 723 script
> mode. Plain `python3 script.py` also still works if the deps happen to be
> installed in your active env.

## Key design decisions (and why)

- **Fetch BOTH speakers, always — and label whose store is whose.** LoCoMo questions
  are about the *shared* dyadic conversation; the answer may live in either speaker's
  memory (especially multi-hop). MemoryBear stores per-`end_user` in the first person,
  so we query both and merge. Each block is wrapped in a `MEMORY STORE: <name>` header,
  and the system prompt states the convention ("inside a store, 'the user' / 'I' = that
  store's owner; any other name = the other person") so the LLM can attribute
  "the user did X" to the right speaker — first-person storage otherwise drops the
  subject's name. This also removes the "question has no name in it" problem entirely.
  (It can't fix cases where the memory extraction itself tangled the perspective.)

- **`sample_id` is the join key, and it is NOT `conv_1`.** LoCoMo's first
  conversation has `sample_id = conv-26` (`conv_2 → conv-30`, … `conv_10 → conv-50`,
  by chronological order in `locomo10.json`). `build_user_map.py` handles this; the
  output uses the real `sample_id`, with an extra `conv_num` field for your eyes.

- **Two scoring tables, both via LoCoMo's `analyze_aggr_acc` (unchanged).** Same
  aggregation code, just a different per-qa metric field.
  - **Table 1 — token-F1 (LoCoMo official).** `eval_question_answering`, untouched:
    cat 4/2/3 token-F1, cat 1 comma-split multi-answer F1, cat 5 = 1.0 only if the
    answer is a refusal (`"no information available"` / `"not mentioned"`). Harsh on
    paraphrase; comparable to the original LoCoMo paper.
  - **Table 2 — LLM-as-judge.** an LLM judges cat 1-4 as CORRECT/WRONG (semantic
    match, robust to date/paraphrase differences); cat 5 keeps the same refusal
    rule. Verdicts are cached in `results/locomo/judge_cache.jsonl` keyed by
    `(category, question, gold, prediction)`, so re-runs are instant and free.
    Comparable to mem0 / Zep. Skip it with `--no-judge`.

  > The two scales are NOT interchangeable — compare F1-to-F1 and judge-to-judge.
  > On conv_1 we saw cat-2 jump 0.54 (F1) → 0.92 (judge): `"May 7, 2023"` vs
  > `"7 May 2023"` is a paraphrase the judge accepts and F1 penalizes.

  Two environment-only shims, **neither touches LoCoMo's eval logic**:
  1. `score.py` stubs the unused `from bert_score import score` (it's imported at
     module top but never called on the QA F1 path) so we don't need
     transformers/torch.
  2. cat-5 items ship only `adversarial_answer` (no `answer`), which would
     `KeyError` inside `eval_question_answering`. `run_predictions.py` copies
     `adversarial_answer → answer` in the prediction record. cat-5 scoring ignores
     the value, so this changes no result; it only prevents a crash. (The original
     `gpt_utils.py` has the same latent bug on this data.)

- **Client-side rerank is an opt-in second retrieval stage (`RERANK_ENABLED=1`).**
  The server keeps ranking (kw+emb hybrid) but only supplies a *deep* candidate pool
  (`TOP_K_LIMIT=50`); selection moves to the client: `rerank.py` scores each
  candidate against the question with gte-rerank-v2 (pointwise cross-encoder) and
  keeps `RERANK_TOP_N`. `data.answer` is ignored on this path — context is rebuilt
  from the winning items' `content` plus the surviving entities' timestamped
  descriptions. Three hard-won details:
  - **Pin by score, not source**: `score >= 0.999` marks server-forced items (the
    `<user-info>` profile). At limit=50 a response can carry 10–20 `ExtractedEntity`
    items with *real* scores — they must compete in the rerank, not be pinned.
  - **Two caches**: raw 50-candidate responses in `raw_fetch_cache.jsonl` (the
    expensive asset — selection-policy experiments replay from it for free) and
    picks/scores in `rerank_cache.jsonl`. Cache keys do NOT include `limit` — use a
    fresh `RESULTS_DIR` whenever you change it, or you'll replay stale responses.
  - **Measured effect is category-dependent** (conv_1, judge noise controlled):
    single-hop/open-domain improve (deeper pool rescues evidence outside the old
    top-10), but multi-hop/enumeration drop sharply — pointwise scoring picks the 10
    items *individually* most similar to the query, not the 10 that jointly cover a
    multi-part answer. Don't adopt it wholesale on enumeration-heavy sets; consider
    a union/RRF fallback with the server order.
  The published run used `TOP_K_LIMIT=50`, `RERANK_TOP_N=10`.

- **Re-judging is decided by the cache *key*, not the verdict.** For cat 1-4,
  `score.py` calls the judge only when `(category, question, gold, prediction)` is
  **not** already a key in `judge_cache.jsonl`; if the key is present it reuses that
  verdict (`0` or `1`) with no LLM call. So an unchanged answer is never re-rolled,
  while `--redo-wrong` / `--refetch` / `--overwrite` change the prediction → new key →
  one fresh judge. This makes re-running `score.py` idempotent and free, and
  `--redo-wrong` monotonic — a *correct* answer keeps its prediction, so it's never
  re-judged and can't regress. (cat 5 is re-derived by the refusal rule every run.)

## What you get / what you don't

- **You get:** per-category + overall accuracy in **two metrics** (token-F1 and
  LLM-judge). Category legend: `1=multi-hop 2=temporal 3=open-domain 4=single-hop
  5=adversarial`.
- **You don't get:** LoCoMo's retrieval *recall* metric. It needs the gold dialog
  ids (`evidence`, e.g. `D1:3`) to overlap with the ids you retrieved — but the
  memory system returns its own memory UUIDs, not dialog ids, so there's nothing to
  overlap. Diagnostic only; not part of the main accuracy numbers.

## Risks to verify with `preflight.py` before a full run

1. **Temporal grounding (cat 2, ~16% of questions).** The recalled context must
   contain the *original* conversation dates (2023…), not just the ingestion
   timestamp. `preflight.py` prints the context and the years it finds. If
   no 2023-era dates appear, cat 2 will largely fail and the fix is on the write
   side (preserve event time), not here.
2. **`search_switch` semantics.** We need it to return retrieved memory fragments
   (so our LLM does the answering under LoCoMo's prompt). If the context looks like
   a finished answer instead of memories, you're testing your answerer, not the
   benchmark setup. `search_switch=2` is expected to return memories.

## Dependencies

Declared per-script via PEP 723 inline metadata (the `# /// script` block at the
top of each file); `uv run` resolves them into an isolated, cached env:

- `run_predictions.py`, `preflight.py` → `requests`
- `score.py` → `regex`, `nltk`, `numpy`, `tqdm` (what LoCoMo's F1 path imports)
- `reproduce.py` → `openai`, `backoff`, `tqdm`, `regex`, `nltk`, `numpy`
- `build_user_map.py` → stdlib only

`bert_score`/`transformers`/`torch` are **not** needed — `score.py` and
`reproduce.py` stub the unused `bert_score` import. Requires `uv` installed; no
conda env, no manual pip.
