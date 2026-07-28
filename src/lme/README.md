# RedBearAI Memory Service × LongMemEval Evaluation Pipeline

> **Just want to reproduce our published results?** No memory-service access needed:
> the memory retrieved for every question ships with this repo in
> `results/lme/memorybear/memorybear_lme_retrieved_memories.json`, and `uv run src/lme/reproduce.py`
> runs reader + judge in one command (any OpenAI-compatible provider) — see the
> [repo root README](../../README.md). Below is the full pipeline, which depends on
> the MemoryBear service and is used for our internal evaluation.

The evaluation data is first written into the memory service question by question
by an internal ingest tool (step 0, which produces the ingest manifests under
`manifests/lme/`). Then: **retrieve the relevant memory** from the memory service,
have a reader LLM **turn the memory into an answer**, and finally score with
qwen3.7-plus as LLM-judge. A three-stage pipeline, all run with `uv run` —
**no conda/venv setup needed** (scripts carry PEP 723 dependency declarations;
uv builds an isolated environment automatically on first run).

```
fetch_mem_hyp.py  ->  generate_hyp_interm.py  ->  evaluate_qa.py
   retrieve           memory -> answer       judge
```

`fetch_mem_hyp.py` is new; `generate_hyp_interm.py` is the new reader stage; `evaluate_qa.py` is adapted from the upstream script of the same name —
the judge prompts and verdict logic are kept verbatim, changed to read `.env`, use
qwen3.7-plus, and use `trials.json` as input/output. `print_metrics.py` /
`count_tokens.py` handle metric aggregation and memory token counting. The three
stages share one artifact, `results/lme/trials.json`: fetch appends records and stores
the memory in `read_response` (leaving `hypothesis` empty), generate reads the memory
and writes the answer into `hypothesis`, evaluate judges the records that have
answers. Every stage is incremental and idempotent — it only processes what the
previous stage just produced and it has not handled yet.

## Prerequisites

1. Install [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
2. The evaluation data (e.g. `data/lme/longmemeval_s_cleaned.json`) has already been
   written into the memory service question by question by the internal ingest tool,
   with the resulting ingest manifests placed under `manifests/lme/`
3. Configure `.env`:

```bash
cd redbear-mem-benchmark
cp .env.example .env   # then fill in real values
```

| Variable               | Description                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `MEMORY_SERVICE_URL`   | Memory service URL                                                                                                                     |
| `MEMORY_API_KEY`       | Memory service API key (sent as `Authorization: Bearer <key>` by default)                                                              |
| `MEMORY_CONFIG_ID`     | Memory config UUID                                                                                                                     |
| `MEMORY_MANIFEST`      | Ingest manifest json (or a directory), decides which case_ids to fetch and each one's end_user_id                                      |
| `MEMORY_SEARCH_SWITCH` | Default `1`                                                                                                                            |
| `LONGMEMEVAL_REF_FILE` | Reference data json; question texts are extracted from here by question_id                                                             |
| `TRIALS_FILE`          | The single artifact: a JSON array with one trial per fetch, shared by all three stages; default `results/lme/trials.json` (gitignored) |
| `READER_MODEL`         | Reader model name for step 2, e.g. `qwen3.7-plus`                                                                                      |
| `READER_API_KEY`       | OpenAI-compatible API key for the reader                                                                                               |
| `READER_API_BASE`      | OpenAI-compatible base url for the reader; defaults to DashScope's compatible-mode endpoint                                            |
| `DASHSCOPE_API_KEY`    | DashScope key for the qwen3.7-plus judge in step 3                                                                                     |

Optional: `MEMORY_READ_PATH` (default `/v1/memory/read/sync`), `MEMORY_API_KEY_HEADER`
(set to e.g. `X-API-Key` if the service expects a custom header), `READER_API_BASE`,
`DASHSCOPE_API_BASE`.

Scripts look for `.env` in the current directory first, then at this repo's root.

## Step 1: retrieve memory from the service (fetch_mem_hyp.py)

**Driven by the ingest manifest**: the manifest produced when data was written into
the memory service (a `cases[]` list, each with `case_id` and `end_user_id`) decides
which cases to fetch and which end_user_id to use for each — no hand-typing
end_user_ids or question_ids. Question texts are still extracted from
`LONGMEMEVAL_REF_FILE` by question_id and POSTed as `message` to
`{MEMORY_SERVICE_URL}/v1/memory/read/internal`, storing the full response in
`read_response` (its `data.answer` is a structured memory dump). **This step only
retrieves memory, it does not generate answers**: `hypothesis` stays empty and is
filled by generate in step 2.

```bash
# fetch every case in the manifest (manifest path from MEMORY_MANIFEST in .env)
uv run src/lme/fetch_mem_hyp.py

# fetch only a few (must be in the manifest)
uv run src/lme/fetch_mem_hyp.py <qid1> <qid2>

# re-test the same question
uv run src/lme/fetch_mem_hyp.py --repeat <qid>

# use a different manifest just for this run (overrides .env)
uv run src/lme/fetch_mem_hyp.py --manifest manifests/lme/FILE_NAME.json
```

`MEMORY_MANIFEST` pointing at a single manifest file = fetch just that batch;
pointing at a directory = merge all manifests in it (when the same case_id appears
multiple times, the end_user_id from the latest `timestamp` wins). Cases with
`status != "ok"` are skipped.

Common options:

- `--manifest PATH`: override `MEMORY_MANIFEST` from `.env`
- `--repeat`: re-test — fetch again even if already fetched, recording a new trial each time (see below)
- `--workers N`: concurrent requests (default 1; ~6 s per question, 4–8 workers speed things up noticeably — mind service rate limits)
- `--include-question-date`: append question_date to the message (may help temporal-reasoning questions)
- `--trials FILE`: override `TRIALS_FILE` from `.env`
- `--max-retries` / `--timeout`: default 3 retries, 120 s timeout

**The single artifact `TRIALS_FILE` (default `results/lme/trials.json`): a JSON array
with one trial per fetch**, self-contained:

```json
{
  "trial_id": "20260616_145247_gpt4_2655b836",   // run-timestamp_question_id, unique
  "run_ts": "20260616_145247",
  "question_id": "...", "question_type": "...", "end_user_id": "...",
  "question": "...", "answer": "...",        // answer is the golden answer
  "hypothesis": null,                        // step 2 generate writes the answer here
  "latency_s": 1.23,
  "read_response": { ...full read API response incl. all intermediate_outputs; data.answer is the memory dump... },
  "reader": null,                            // step 2 generate writes {"model": ...}
  "autoeval_label": null                     // filled when judged in step 3
}
```

- **The same question can be tested repeatedly, keeping every run on record**: use
  `--repeat <case_id>` to fetch again — it appends a new trial (different
  `trial_id`); existing records are never overwritten. trials.json is the one big
  document holding the full history.
- **Resumable**: without `--repeat`, question_ids already present in trials.json are
  skipped and only unfetched ones are retrieved; failures are listed at the end —
  just re-run to retry.
- trials.json lives under `results/lme/` and is gitignored (the directory is created
  automatically; use `git add -f` if you want to share one).

> end_user_id comes from the manifest per case, so each can differ and batch fetches
> never hit the wrong user. If a case_id is in the manifest but not in
> `LONGMEMEVAL_REF_FILE` (no question text available), it is warned about and skipped.

## Step 2: the reader turns memory into an answer (generate_hyp_interm.py)

Fetch only retrieved a blob of memory. This step sends
`question + question_type + that memory` to a reader LLM (OpenAI-compatible API)
with a question-type-specific prompt, asking it to produce a concise answer
**grounded only in the memory**, written into `hypothesis`, marked with `reader`,
and with `autoeval_label` reset to `null` (the answer changed, so any earlier
judgement is stale — step 3 re-judges).

The memory fed to the reader is built from the structured
`read_response.data.intermediate_outputs`: each item contributes its `content`
(joined together these are verbatim identical to the `data.answer` blob), and
each `ExtractedEntity` item **additionally** attaches the **timestamped** detail
from its `data.description` (e.g. `[2023-04-10T17:15:00Z] ...`, with `；` split
into one line each). `Statement` / `MemorySummary` items have no description and
are left unchanged. So the reader sees ≈ the original memory dump + timestamped
details under each entity block, which helps on temporal-reasoning /
knowledge-update / multi-session questions that require ordering events in time
or synthesizing across sessions. This is the reading protocol behind the
published results (and the one `reproduce.py` replays).

```bash
# defaults to TRIALS_FILE from .env
uv run src/lme/generate_hyp_interm.py

# or specify the trials file explicitly
uv run src/lme/generate_hyp_interm.py results/lme/trials.json

# force-regenerate trials that already have a hypothesis
uv run src/lme/generate_hyp_interm.py results/lme/trials.json --force
```

- **The input is always `read_response.data.intermediate_outputs`, never
  `hypothesis`** — so even after `hypothesis` has been overwritten with a
  generated answer, re-running never feeds the answer back as input; idempotent
  and safe.
- Only trials **without a `reader` marker** (not yet generated) are processed;
  already-generated ones are skipped unless `--force`. So when testing batch by
  batch, each run only generates the new batch.
- The prompt requires the reader to **explicitly say "cannot be answered"** rather
  than guess when the memory lacks the information — this both handles the `_abs`
  unanswerable questions correctly and suppresses hallucination.
- Reader model / key / base url come from `READER_MODEL` / `READER_API_KEY` /
  `READER_API_BASE` in `.env` (`temperature=0`, `max_tokens=512`).

## Step 3: qwen3.7-plus judging (evaluate_qa.py)

The judging logic (prompt templates, `n=1/temperature=0/max_tokens=10`, `'yes' in`
verdict) is verbatim identical to the original `evaluate_qa.py`; qwen3.7-plus is called
through DashScope's OpenAI-compatible API. The input is `TRIALS_FILE` directly
(trials carry question / answer / question_type / hypothesis, **no ref file
needed**). Only trials with a non-empty `hypothesis` and a `null` `autoeval_label`
are judged; trials that haven't been through generate (empty `hypothesis`) are
reported and skipped.

```bash
# defaults to TRIALS_FILE from .env
uv run src/lme/evaluate_qa.py qwen3.7-plus

# or specify the trials file explicitly
uv run src/lme/evaluate_qa.py qwen3.7-plus results/trials.json

# additionally export an original-style per-question eval-results jsonl
uv run src/lme/evaluate_qa.py qwen3.7-plus --legacy
```

It only judges trials with a non-empty `hypothesis` whose `autoeval_label` is still
`null`, writing results back; **already-judged trials are untouched** — so if you
test 10 at a time, each evaluate run only judges the new batch instead of re-judging
everything. Idempotent, history preserved.

Output:

- **Written back into `TRIALS_FILE`**: new trials get `autoeval_label` filled with `{"model": ..., "label": ...}`
- Accuracy printed to the terminal: deduplicated by **the latest trial per question**
  (`--repeat` history isn't double-counted), overall and per question_type
- `--legacy`: also writes `results/lme/eval-results-<model>.jsonl`, one question per line
  (latest trial), in the original `{question_id, hypothesis, autoeval_label}` format

Entries whose question_id ends with `_abs` automatically use the "unanswerable
question" judging template, same as the original.

## Common issues

- **Missing config: ...**: `.env` is incomplete or wasn't found; the error message shows which .env was actually loaded
- **API returned code N**: the memory service returned a non-zero code; retried automatically, just re-run for the ones that ultimately failed
- **Missing config: READER_MODEL/READER_API_KEY**: reader model / key for step 2 not configured
- **DASHSCOPE_API_KEY is not set**: judge key for step 3 not configured
- **Trials file not found**: nothing fetched yet, or `TRIALS_FILE` path is wrong
- **judge says "have no hypothesis yet"**: those trials haven't been through generate; run step 2 first
- Step 2 only generates trials **without a `reader` marker**; to regenerate with a different reader model, use `--force`
- Step 3 only judges trials whose `autoeval_label` is `null`; to re-judge existing ones with a different judge model, first reset those trials' `autoeval_label` to `null` (or delete them)
- Step 3 only prints question_types that have data; when evaluating a few questions the other types won't show (and there's no numpy nan warning)
- **Must use `uv run`**: scripts carry PEP 723 dependencies; plain `python3` fails with missing `backoff` / `openai`
