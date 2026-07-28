#!/usr/bin/env python3
"""Shared config for the LoCoMo eval scripts: .env loading + data/result paths.

Paths are configurable via .env (see .env.example at the repo root):
    LOCOMO_JSON   dataset file scripts read questions/speakers from
    MANIFEST_DIR  ingestion manifests (build_user_map.py input)
    USER_MAP      conv -> end_user_id map (build_user_map.py output,
                  run_predictions.py input)
    RESULTS_DIR   predictions + caches + scores live here

Every key is optional -- a missing key falls back to the default below, so the
scripts run unchanged without a .env. Relative paths in .env are resolved
against this directory (src/locomo/). Real OS environment variables override
.env values. The .env file is searched in the current working directory, then
the repo root.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # src/locomo/
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))      # repo root

# keys an OS environment variable may override even when absent from .env
_ENV_KEYS = [
    "MEM_SERVICE_URL", "MEM_API_KEY", "SEARCH_SWITCH", "MEM_CONFIG_ID",
    "TOP_K_LIMIT", "MEM_SKIP_SUMMARY",
    "RERANK_ENABLED", "RERANK_MODEL", "RERANK_URL", "RERANK_API_KEY", "RERANK_TOP_N",
    "MEM_AUTH_HEADER", "MEM_AUTH_PREFIX", "LLM_URL", "LLM_API_KEY", "LLM_MODEL",
    "LOCOMO_JSON", "MANIFEST_DIR", "USER_MAP", "RESULTS_DIR",
]

_ENV_CANDIDATES = [
    os.path.join(os.getcwd(), ".env"),
    os.path.join(REPO_ROOT, ".env"),
]


def load_env(path=None):
    env = {}
    candidates = [path] if path else _ENV_CANDIDATES
    for cand in candidates:
        if cand and os.path.exists(cand):
            for line in open(cand):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
            break
    # OS env overrides .env
    for k in list(env.keys()) + _ENV_KEYS:
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def require(env, *keys):
    missing = [k for k in keys if not env.get(k)]
    if missing:
        sys.exit("Missing required env vars: %s (set them in .env)" % ", ".join(missing))


def _path(env, key, default):
    v = env.get(key)
    if not v:
        return default
    return v if os.path.isabs(v) else os.path.normpath(os.path.join(HERE, v))


_env = load_env()
LOCOMO_JSON = _path(_env, "LOCOMO_JSON",
                    os.path.join(REPO_ROOT, "data/locomo/locomo10.json"))
MANIFEST_DIR = _path(_env, "MANIFEST_DIR",
                     os.path.join(REPO_ROOT, "manifests/locomo"))
USER_MAP = _path(_env, "USER_MAP", os.path.join(HERE, "user_map.json"))
RESULTS_DIR = _path(_env, "RESULTS_DIR", os.path.join(REPO_ROOT, "results/locomo"))
