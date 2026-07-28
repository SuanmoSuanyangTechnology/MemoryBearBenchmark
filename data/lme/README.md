# data/lme — the LongMemEval dataset

The dataset is **not** redistributed in this repo. Download the cleaned release
from Hugging Face into this directory:

```bash
uv run data/lme/download_lme.py            # -> data/lme/longmemeval_s_cleaned.json
uv run data/lme/download_lme.py --force    # re-download / overwrite
```

`longmemeval_s_cleaned.json` (~277 MB) holds the 500 questions with their
~115k-token haystack chat histories — the file our published run used. The
script validates the file (valid JSON, exactly 500 question records) before
replacing any local copy.

**You do not need this for quick reproduction** — `src/lme/reproduce.py` runs
entirely from the released memories under `results/lme/memorybear/`. The dataset is used
by the full pipeline: ingestion writes the chat histories into the MemoryBear
service, and `src/lme/fetch_mem_hyp.py` extracts question texts from it
(`LONGMEMEVAL_REF_FILE` in `.env`).

Downloaded files in this directory are gitignored.
