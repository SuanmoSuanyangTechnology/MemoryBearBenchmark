# data/locomo — the LoCoMo dataset

The dataset is **not** redistributed in this repo (it is CC BY-NC 4.0,
© Snap Inc.). Download it from the official repository into this directory:

```bash
uv run data/locomo/download_locomo.py            # -> data/locomo/locomo10.json
uv run data/locomo/download_locomo.py --force    # re-download / overwrite
```

`locomo10.json` (~2.8 MB) holds 10 very-long-term two-person conversations with
their QA annotations. The script validates the file (valid JSON, exactly 10
conversation records) before replacing any local copy.

**You do not need this for quick reproduction** — `src/locomo/reproduce.py`
runs entirely from the released memories under `results/locomo/memorybear/`. The dataset
is read by the full pipeline (`src/locomo/run_predictions.py`, `score.py`;
path configurable via `LOCOMO_JSON` in `.env`).

Downloaded files in this directory are gitignored.
