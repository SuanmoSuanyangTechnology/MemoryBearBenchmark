# MemoryBear Memory Benchmarks

**English** | [中文](README.zh-CN.md)

**MemoryBear** is a next-generation AI memory system developed by **RedBear AI**. Its core breakthrough lies in moving beyond the limitations of traditional "static knowledge storage". Inspired by the cognitive mechanisms of biological brains, MemoryBear builds an intelligent knowledge-processing framework that spans the full lifecycle of **perception → extraction → association → forgetting**.

Unlike traditional memory tools that treat knowledge as static data to be retrieved, MemoryBear emulates the hippocampus's memory encoding, the neocortex's knowledge consolidation, and synaptic pruning-based forgetting — enabling knowledge to dynamically evolve with life-like properties, and shifting the relationship between AI and users from passive lookup to proactive cognitive assistance.

This repository records our evaluations of MemoryBear on two public long-term
memory benchmarks and provides everything needed to reproduce the results: the
evaluation code we ran, the per-question memories MemoryBear retrieved, and a
one-command script per benchmark that replays the answer-generation and
scoring half of the pipeline.

## Results

| Benchmark | Questions | Headline result |
| --------- | --------: | --------------- |
| **LongMemEval** (ICLR 2025) | 500 | **95.0%** LLM-judge accuracy |
| **LoCoMo** (ACL 2024) | 1,986 | **91.5%** LLM-judge accuracy · **0.675** token-F1 |

Both evaluations use the benchmark's own judge protocol and scoring code
unmodified. Full per-type/per-category tables, the exact setup, and the
released artifacts: **[docs/results.md](docs/results.md)**. The released
per-question artifacts are also mirrored as a Hugging Face dataset:
[redbearai/MemoryBear_eval_result](https://huggingface.co/datasets/redbearai/MemoryBear_eval_result).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and an LLM API key
(any OpenAI-compatible provider works, endpoint set via `LLM_BASE_URL`).

```bash
cp .env.example .env               # fill in LLM_API_KEY

uv run src/lme/reproduce.py        # LongMemEval (500 questions)
uv run src/locomo/reproduce.py     # LoCoMo (1,986 questions)
```

Each script runs directly on the released retrieved memories
(`results/<benchmark>/memorybear/*_retrieved_memories.json(.gz)` — the exact
per-question context MemoryBear returned in our runs): it feeds them to a
reader LLM and scores the answers with the original benchmark protocol — no
access to the MemoryBear service needed. Both are resumable, print per-category tables at
the end, and accept `--reader-model` / `--judge-model` / `--workers` /
`--force`.

To also inspect the source datasets or run the full internal pipeline:

```bash
uv run data/lme/download_lme.py        # LongMemEval dataset (~277 MB, from Hugging Face)
uv run data/locomo/download_locomo.py  # LoCoMo dataset (~2.8 MB, from GitHub)
```

## Documentation

| Doc | Contents |
| --- | -------- |
| [docs/pipeline.md](docs/pipeline.md) | The four-stage pipeline (ingest → retrieve → answer → judge) and the design principles |
| [docs/results.md](docs/results.md) | Full result tables, evaluation setup, released artifacts, and how to verify the numbers |
| [src/lme/README.md](src/lme/README.md) | LongMemEval pipeline deep-dive (fetch → read → judge) |
| [src/locomo/README.md](src/locomo/README.md) | LoCoMo pipeline deep-dive (two-store fetch, rerank, LoCoMo scoring) |

## Repository layout

```
├── docs/               pipeline & results documentation
├── src/
│   ├── lme/            LongMemEval evaluation code + reproduce.py
│   └── locomo/         LoCoMo evaluation code + reproduce.py
│       └── task_eval/  LoCoMo's official scoring code (vendored verbatim)
├── data/
│   ├── lme/            dataset download script (dataset itself gitignored)
│   └── locomo/         dataset download script (dataset itself gitignored)
├── results/
│   ├── lme/            released artifacts per system (memorybear/, baselines)
│   └── locomo/         released artifacts per system (memories gzipped)
├── manifests/          internal ingestion manifests (placeholders only)
└── .env.example        one key (LLM_API_KEY) is enough to reproduce
```

## Notices

The benchmark-derived components are licensed from their upstream projects —
LongMemEval (MIT) and LoCoMo (CC BY-NC 4.0) — see [NOTICE](NOTICE) for
details. The benchmark datasets are not redistributed here — the scripts
under `data/` download them from their official sources.

## Contact

Questions about MemoryBear or the evaluations: the RedBear AI team.
