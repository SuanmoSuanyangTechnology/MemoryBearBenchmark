# The evaluation pipeline

Both benchmarks run through the same four-stage pipeline:

```
1. ingest                2. retrieve               3. answer               4. judge
write each conversation  query the memory service  reader LLM turns        the benchmark's own
into MemoryBear, with    per question, record the  question + memory       protocol, unmodified
original timestamps      retrieved memory context  into a concise answer
                                   │                        ▲
                                   └────────────────────────┘
                     results/<benchmark>/memorybear/*_retrieved_memories.json(.gz)
                     what stage 2 retrieved is released per question and
                     is everything stages 3–4 need
```

- **Stage 1 — ingest.** Every conversation is written into the MemoryBear
  service with its original timestamps, producing ingest manifests (under
  `manifests/`, placeholders only — they contain internal `end_user_id`s).
  MemoryBear then does its own perception → extraction → association →
  forgetting processing; nothing benchmark-specific is injected.
- **Stage 2 — retrieve.** For each benchmark question we call the memory
  search API (`search_switch=2`, which returns memory fragments rather than a
  finished answer) and record the retrieved context. What is retrieved per
  question is exactly what we release.
- **Stage 3 — answer.** A reader LLM turns `question + retrieved memory` into
  a concise answer, with per-question-type (LME) or per-category (LoCoMo)
  prompt guidance. The reader never sees the original conversation — only the
  memory.
- **Stage 4 — judge.** The benchmark's own protocol, unmodified: LongMemEval's
  original judge prompts; LoCoMo's official token-F1 plus an LLM judge
  (abstention rule for adversarial questions).

## One-command reproduction

For every question we release the memory MemoryBear retrieved. One script per
benchmark replays stages 3–4 from those files — stages 1–2 need the MemoryBear
service, but their output ships with the repo, so reproduction needs nothing
beyond an LLM API key:

```bash
cp .env.example .env               # fill in LLM_API_KEY
uv run src/lme/reproduce.py        # LongMemEval  (500 questions)
uv run src/locomo/reproduce.py     # LoCoMo       (1,986 questions)
```

Both scripts use the same reader prompts, sampling parameters, and judge
protocol as the published runs, are resumable, and print per-category tables at
the end. Numbers reproduced with a different reader/judge model will differ
somewhat from ours (see [results.md](results.md) for the models we used).

## Per-benchmark pipelines

The full pipelines (from ingestion) are in the repo and documented in depth.
The ingest and retrieve stages require MemoryBear service access:

| Benchmark | Pipeline | Deep-dive doc |
| --------- | -------- | ------------- |
| LongMemEval | `fetch_mem_hyp.py → generate_hyp_interm.py → evaluate_qa.py`, sharing one incremental `trials.json` artifact | [src/lme/README.md](../src/lme/README.md) |
| LoCoMo | `build_user_map.py → preflight.py → run_predictions.py → score.py`, with an optional client-side rerank stage | [src/locomo/README.md](../src/locomo/README.md) |

LoCoMo specifics worth knowing: every question queries **both** speakers'
memory stores and merges them under labeled `MEMORY STORE: <name>` headers
(LoCoMo questions are about the shared conversation, and MemoryBear stores
per-person in the first person); the published run fetched 50 candidates per
store and reranked them client-side down to 10 with gte-rerank-v2.

## Shared design principles

- **Original scoring, verbatim.** LongMemEval judge prompts are used unchanged;
  LoCoMo's scoring code is vendored verbatim in
  [src/locomo/task_eval/](../src/locomo/task_eval/) and imported as-is.
- **Per-question artifacts.** Every stage's output is released per question
  (memories → hypotheses → judgments → metrics), so any single number can be
  traced back to the exact memory and answer behind it.
- **Idempotent, resumable stages.** Every script processes only what earlier
  stages produced and it hasn't handled yet; interrupting and re-running is
  always safe.
- **Zero environment setup.** Every script carries PEP 723 inline dependency
  metadata; `uv run` builds an isolated environment on first run — no conda,
  no venv, no pip install.
