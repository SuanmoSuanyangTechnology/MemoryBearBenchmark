# Results

MemoryBear's published results on both benchmarks, the setup that produced
them, and how to verify the numbers yourself.

## Summary

| Benchmark                               | Questions | LLM-judge accuracy |  Token-F1 |
| --------------------------------------- | --------: | -----------------: | --------: |
| [LongMemEval](#longmemeval) (ICLR 2025) |       500 |          **95.0%** |         — |
| [LoCoMo](#locomo) (ACL 2024)            |     1,986 |          **91.5%** | **0.675** |

Both evaluations use the benchmark's own judge protocol and scoring code
unmodified. Per-question artifacts live under `results/<benchmark>/<system>/`
— `memorybear/`.

## Setup

|            | LongMemEval                                                         | LoCoMo                                                                                                                          |
| ---------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Dataset    | `longmemeval_s_cleaned.json` (500 questions, ~115k-token histories) | `locomo10.json` (10 conversations, 1,986 questions)                                                                             |
| Ingestion  | full timestamped chat history per question, internal tool           | all 10 conversations, one memory store per speaker                                                                              |
| Retrieval  | memory search API, `search_switch=2` (returns memory fragments)     | both speakers' stores queried per question; top-50 candidates per store reranked client-side (gte-rerank-v2) to 10, then merged |
| Reader     | qwen3.7-plus, temperature 0                                         | qwen3.7-plus, temperature 0, max_tokens 64                                                                                      |
| Judge      | qwen3.7-plus, original LongMemEval prompts (verbatim)               | qwen3.7-plus with the J-score-style judge prompt in `score.py`; token-F1 via LoCoMo's own `eval_question_answering`             |
| Abstention | `_abs` questions use the original unanswerable-question template    | category 5 counts as correct only on an explicit refusal, under both metrics                                                    |

MemoryBear ran as-is — no benchmark-specific tuning was injected at ingestion
or retrieval time; the same service configuration served both benchmarks.

## LongMemEval

LLM-judge accuracy over all 500 questions (original LongMemEval judge prompts):

| Question type             |   # | Accuracy  |
| ------------------------- | --: | --------- |
| single-session-user       |  70 | 100.0%    |
| single-session-preference |  30 | 100.0%    |
| knowledge-update          |  78 | 98.7%     |
| multi-session             | 133 | 94.0%     |
| temporal-reasoning        | 133 | 94.0%     |
| single-session-assistant  |  56 | 85.7%     |
| **Overall**               | 500 | **95.0%** |

The memory retrieved per question averages **~707 tokens** (vs. feeding a
model the full ~115k-token chat history), with a median retrieval latency of
**~0.5 s**.

### Comparison with other memory systems

Evaluated on the full 500-question LongMemEval set; accuracy is determined by
an LLM judge.

| System      | single-session-preference | single-session-assistant | temporal-reasoning | multi-session | knowledge-update | single-session-user | overall   |
| ----------- | ------------------------- | ------------------------ | ------------------ | ------------- | ---------------- | ------------------- | --------- |
| MemoryBear  | **100%**                  | 85.71%                   | **93.98%**         | **93.98%**    | **98.72%**       | **100%**            | **95.0%** |
| MemOS       | 86.67%                    | **92.86%**               | 81.95%             | 80.45%        | 94.87%           | 98.57%              | 87.4%     |
| Memobase    | 78.40%                    | 22.51%                   | 72.13%             | 63.56%        | 87.05%           | 91.00%              | 69.65%    |
| Mem0        | 88.20%                    | 25.98%                   | 68.57%             | 59.99%        | 64.67%           | 81.20%              | 63.86%    |
| Zep         | 52.23%                    | 72.75%                   | 51.40%             | 45.03%        | 72.17%           | 91.04%              | 61.51%    |
| Supermemory | 88.20%                    | 57.15%                   | 42.14%             | 50.00%        | 53.47%           | 84.00%              | 56.31%    |
| MIRIX       | 52.26%                    | 61.72%                   | 24.28%             | 28.57%        | 50.98%           | 71.39%              | 42.02%    |
| MemU        | 75.14%                    | 19.05%                   | 16.43%             | 40.00%        | 39.79%           | 65.80%              | 37.07%    |

Released artifacts:

| File                                                                                                                              | Contents                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [results/lme/memorybear/memorybear_lme_retrieved_memories.json](../results/lme/memorybear/memorybear_lme_retrieved_memories.json) | Per question: the memory MemoryBear retrieved, plus question / golden answer / retrieval latency |
| [results/lme/memorybear/memorybear_lme_hypotheses.json](../results/lme/memorybear/memorybear_lme_hypotheses.json)                 | + the reader's generated answer                                                                  |
| [results/lme/memorybear/memorybear_lme_judged.json](../results/lme/memorybear/memorybear_lme_judged.json)                         | + LLM-judge label and per-question metrics                                                       |
| [results/lme/memorybear/memorybear_lme_metrics.json](../results/lme/memorybear/memorybear_lme_metrics.json)                       | Aggregated metrics: overall / by type / by question                                              |
| [results/lme/memorybear/memorybear_lme_results.xlsx](../results/lme/memorybear/memorybear_lme_results.xlsx)                       | Summary spreadsheet                                                                              |

## LoCoMo

Both tables cover all 1,986 questions and are produced by LoCoMo's own
aggregation code, unmodified. **LLM-judge accuracy** grades semantic
correctness (comparable to how mem0 / Zep report LoCoMo); **token-F1** is
LoCoMo's official lexical metric (comparable to the original paper; it
penalizes any paraphrase, so the two scales are not interchangeable).

| Category    |    # | LLM-judge acc. | Token-F1  |
| ----------- | ---: | -------------- | --------- |
| single-hop  |  841 | 92.3%          | 0.665     |
| multi-hop   |  282 | 90.8%          | 0.547     |
| temporal    |  321 | 91.6%          | 0.519     |
| open-domain |   96 | 74.0%          | 0.413     |
| adversarial |  446 | 94.4%          | 0.944     |
| **Overall** | 1986 | **91.5%**      | **0.675** |

### Comparison with other memory systems

Evaluated on the full LoCoMo benchmark (1,986 questions across 10
conversations); accuracy is determined by an LLM judge. The system baselines
cover the 1,540 non-adversarial questions, so their `adversarial` cells are
empty and their overall scores are computed over the remaining four
categories.

| System      | single-hop | multi-hop  | temporal-reasoning | open-domain | adversarial | overall    | overall F1 |
| ----------- | ---------- | ---------- | ------------------ | ----------- | ----------- | ---------- | ---------- |
| MemoryBear  | **92.27%** | **90.78%** | **91.59%**         | **73.96%**  | 94.39%      | **91.54%** | **67.49**  |
| MemOS       | 89.89%     | 77.30%     | 81.93%             | 63.54%      | –           | 84.29%     | 38.44      |
| Mem0        | 80.98%     | 84.40%     | 88.16%             | 73.96%      | –           | 82.66%     | 48.74      |
| Memobase    | 71.66%     | 61.42%     | 77.14%             | 51.53%      | –           | 69.68%     | 50.18      |
| MIRIX       | 66.86%     | 51.55%     | 65.11%             | 45.47%      | –           | 62.29%     | 28.10      |
| Zep         | 64.91%     | 49.51%     | 52.08%             | 32.33%      | –           | 57.39%     | 41.23      |
| MemU        | 65.01%     | 59.96%     | 25.75%             | 48.50%      | –           | 54.87%     | 35.15      |
| Supermemory | 65.95%     | 48.56%     | 30.18%             | 41.39%      | –           | 53.72%     | 34.87      |

Released artifacts:

| File                                                                                                                                                | Contents                                                                                                                      |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [results/locomo/memorybear/memorybear_locomo_retrieved_memories.json.gz](../results/locomo/memorybear/memorybear_locomo_retrieved_memories.json.gz) | Per question: the merged two-store context MemoryBear retrieved, plus question / golden answer / speakers / retrieval latency |
| [results/locomo/memorybear/memorybear_locomo_hypotheses.json](../results/locomo/memorybear/memorybear_locomo_hypotheses.json)                       | Per question: the reader's generated answer (join back to the memories via `question_id`)                                     |
| [results/locomo/memorybear/memorybear_locomo_judged.json](../results/locomo/memorybear/memorybear_locomo_judged.json)                               | + LLM-judge label, token-F1 and per-question lexical metrics                                                                  |
| [results/locomo/memorybear/memorybear_locomo_metrics.json](../results/locomo/memorybear/memorybear_locomo_metrics.json)                             | Aggregated metrics: overall / by category / by conversation / by question                                                     |
| [results/locomo/memorybear/memorybear_locomo_results.xlsx](../results/locomo/memorybear/memorybear_locomo_results.xlsx)                             | Summary spreadsheet                                                                                                           |

## Verifying these numbers

The released `*_retrieved_memories` files contain everything stages 3–4 need
(see [pipeline.md](pipeline.md)); one script per benchmark replays them:

```bash
cp .env.example .env               # fill in LLM_API_KEY
uv run src/lme/reproduce.py        # 2 API calls per question × 500
uv run src/locomo/reproduce.py     # ≤2 API calls per question × 1,986
```

Options (both scripts): `--reader-model` / `--judge-model` (default: the
`LLM_MODEL` env var, else `gpt-4o`; any OpenAI-compatible endpoint via
`LLM_BASE_URL`), `--workers N`, `--force`, `--quiet` (suppress the live
per-question judge lines). Interrupted runs resume where they left off.

Two caveats when comparing to the tables above:

1. **The reader model matters.** Our published numbers used qwen3.7-plus as
   the reader; a different reader produces somewhat different answers and
   therefore somewhat different scores. To match our setup, point
   `LLM_BASE_URL` at DashScope's compatible-mode endpoint and pass
   `--reader-model qwen3.7-plus`.
2. **LLM judges are not perfectly deterministic.** Even at temperature 0,
   verdicts can flip on a handful of borderline questions between runs;
   expect the overall numbers to land within a fraction of a percent.

## References

```bibtex
@article{wu2024longmemeval,
  title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author={Di Wu and Hongwei Wang and Wenhao Yu and Yuwei Zhang and Kai-Wei Chang and Dong Yu},
  year={2024},
  eprint={2410.10813},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2410.10813},
}

@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```
