# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "requests>=2.31",
#     "tqdm>=4.66",
# ]
# ///
"""Fetch hypotheses for LongMemEval questions from the RedBearAI memory service.

Which questions to fetch and the per-question end_user_id are both driven by an
ingest manifest (the json produced when data is written into the memory
service: a `cases` list of {case_id, end_user_id, status, ...}). No need to
type question_ids or end_user_ids by hand.

All other request parameters come from a .env file (see .env.example at the
repo root):
    MEMORY_SERVICE_URL, MEMORY_API_KEY, MEMORY_CONFIG_ID,
    MEMORY_SEARCH_SWITCH (default "1"),
    MEMORY_MANIFEST (ingest manifest json, or a directory of them; gives
    case_id -> end_user_id),
    LONGMEMEVAL_REF_FILE (reference json that question texts are extracted
    from by question_id), TRIALS_FILE (output, default results/trials.json)

For each case_id in the manifest, the question text is looked up in
LONGMEMEVAL_REF_FILE and sent as `message` to
POST {MEMORY_SERVICE_URL}/v1/memory/read/internal under that case's end_user_id;
the full response (whose data.answer is a structured memory dump) is stored as
read_response. This is the first of a three-stage pipeline
(fetch -> generate_hyp_interm.py -> evaluate_qa.py): fetch leaves `hypothesis`
null; generate_hyp_interm.py reads read_response.data.intermediate_outputs and writes the reader
LLM's answer into `hypothesis`; evaluate_qa.py judges it.

Usage:
    # fetch every case in the manifest (manifest path from .env)
    uv run fetch_mem_hyp.py

    # fetch only specific cases (must be in the manifest)
    uv run fetch_mem_hyp.py <qid1> <qid2>

    # point at a specific manifest for this run (overrides .env)
    uv run fetch_mem_hyp.py --manifest manifests/FILE_NAME.json

    # re-test (fetch again even if already done) — records a new trial
    uv run fetch_mem_hyp.py --repeat <qid>

The single output is TRIALS_FILE (default results/trials.json), a JSON array
where each fetch appends one self-contained trial: trial_id, run_ts,
question_id, question_type, end_user_id, question, question_date, answer, the
full read response (read_response), a null hypothesis (generate_hyp_interm.py fills it), a null
reader marker (generate_hyp_interm.py sets it), latency_s, and a null autoeval_label
(evaluate_qa.py fills it). One fetch = one trial. By default a
question already present in trials.json is skipped; --repeat fetches it again
and appends a NEW trial, so the full history is kept and nothing is
overwritten.
"""

import argparse
import glob
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm


def write_json_atomic(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_env_file(path=None):
    candidates = [path] if path else [
        os.path.join(os.getcwd(), '.env'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'),
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            for line in open(cand):
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return cand
    return None


def load_manifest(path):
    """Read ingest manifest(s); return {case_id: end_user_id} for ok cases.

    `path` may be a single manifest json or a directory of them. When the same
    case_id appears in multiple manifests, the one with the latest top-level
    `timestamp` wins (later ingests supersede earlier ones).
    """
    files = sorted(glob.glob(os.path.join(path, '*.json'))) if os.path.isdir(path) else [path]
    docs = []
    for fname in files:
        doc = json.load(open(fname))
        docs.append((doc.get('timestamp', ''), fname, doc))
    docs.sort(key=lambda x: x[0])
    qid2user = {}
    for _, fname, doc in docs:
        for case in doc.get('cases', []):
            if case.get('status') != 'ok':
                print('Warning: case {} in {} has status {!r}, skipping.'.format(
                    case.get('case_id'), os.path.basename(fname), case.get('status')))
                continue
            qid2user[case['case_id']] = case['end_user_id']
    return qid2user


def build_headers(api_key):
    header = os.getenv('MEMORY_API_KEY_HEADER', 'Authorization')
    if header == 'Authorization' and not api_key.lower().startswith('bearer '):
        value = 'Bearer ' + api_key
    else:
        value = api_key
    return {header: value, 'Content-Type': 'application/json'}


def fetch_one(session, url, headers, payload, timeout, max_retries):
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            if body.get('code') != 0:
                raise RuntimeError('API returned code {}: {}'.format(body.get('code'), body.get('msg')))
            if not isinstance((body.get('data') or {}).get('answer'), str):
                raise RuntimeError('API response has no data.answer field')
            return body
        except Exception as err:
            last_err = err
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 30))
    raise last_err


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('question_ids', nargs='*',
                        help='case_ids to fetch; omit to fetch every case in the manifest')
    parser.add_argument('--manifest', default=None,
                        help='ingest manifest json or directory (default: MEMORY_MANIFEST from .env)')
    parser.add_argument('--repeat', action='store_true',
                        help='re-fetch selected questions even if already done; records a new trial each time')
    parser.add_argument('--trials', default=None,
                        help='output trials json (default: TRIALS_FILE from .env, else results/trials.json)')
    parser.add_argument('--env-file', default=None, help='path to .env (default: ./.env, then repo root)')
    parser.add_argument('--workers', type=int, default=1, help='concurrent requests (default: 1)')
    parser.add_argument('--timeout', type=float, default=120.0, help='per-request timeout in seconds')
    parser.add_argument('--max-retries', type=int, default=3, help='attempts per question before giving up')
    parser.add_argument('--include-question-date', action='store_true',
                        help='prepend question_date to the message sent to the memory service')
    args = parser.parse_args()

    env_used = load_env_file(args.env_file)

    service_url = os.getenv('MEMORY_SERVICE_URL')
    api_key = os.getenv('MEMORY_API_KEY')
    config_id = os.getenv('MEMORY_CONFIG_ID')
    manifest_path = args.manifest or os.getenv('MEMORY_MANIFEST')
    search_switch = os.getenv('MEMORY_SEARCH_SWITCH', '1')
    read_path = os.getenv('MEMORY_READ_PATH', '/v1/memory/read/sync')
    ref_file = os.getenv('LONGMEMEVAL_REF_FILE')
    trials_file = args.trials or os.getenv('TRIALS_FILE') or 'results/trials.json'
    missing = [name for name, value in [('MEMORY_SERVICE_URL', service_url),
                                        ('MEMORY_API_KEY', api_key),
                                        ('MEMORY_CONFIG_ID', config_id),
                                        ('MEMORY_MANIFEST', manifest_path),
                                        ('LONGMEMEVAL_REF_FILE', ref_file)] if not value]
    if missing:
        sys.exit('Missing config: {}. Set them in .env (loaded: {}).'.format(
            ', '.join(missing), env_used or 'no .env found'))
    if not os.path.exists(manifest_path):
        sys.exit('Manifest not found: {}'.format(manifest_path))
    url = service_url.rstrip('/') + read_path

    qid2user = load_manifest(manifest_path)
    print('Loaded {} case_id -> end_user_id mappings from {}.'.format(len(qid2user), manifest_path))
    if not qid2user:
        sys.exit('No ok cases found in manifest {}.'.format(manifest_path))

    references = json.load(open(ref_file))
    qid2entry = {entry['question_id']: entry for entry in references}

    if args.question_ids:
        unknown = [qid for qid in args.question_ids if qid not in qid2user]
        if unknown:
            sys.exit('question_ids not in manifest {}: {}'.format(manifest_path, ', '.join(unknown)))
        selected_qids = list(args.question_ids)
    else:
        selected_qids = list(qid2user.keys())

    no_text = [qid for qid in selected_qids if qid not in qid2entry]
    if no_text:
        print('Warning: {} manifest case(s) not found in {}, skipping: {}'.format(
            len(no_text), ref_file, ', '.join(no_text[:10]) + ('...' if len(no_text) > 10 else '')))
    selected = [qid2entry[qid] for qid in selected_qids if qid in qid2entry]

    run_ts = time.strftime('%Y%m%d_%H%M%S')
    trials = json.load(open(trials_file)) if os.path.isfile(trials_file) else []
    trial_ids = {t['trial_id'] for t in trials}
    done = {t['question_id'] for t in trials}

    if args.repeat:
        todo = selected
    else:
        todo = [entry for entry in selected if entry['question_id'] not in done]
    skipped = len(selected) - len(todo)
    if skipped:
        print('Skipping {} questions already in {} (use --repeat to re-fetch).'.format(skipped, trials_file))

    def new_trial_id(qid):
        base = '{}_{}'.format(run_ts, qid)
        tid, n = base, 1
        while tid in trial_ids:
            n += 1
            tid = '{}_{}'.format(base, n)
        trial_ids.add(tid)
        return tid

    headers = build_headers(api_key)
    session = requests.Session()
    out_lock = threading.Lock()
    failures = []
    trials_dir = os.path.dirname(trials_file)
    if trials_dir:
        os.makedirs(trials_dir, exist_ok=True)

    def work(entry):
        message = entry['question']
        if args.include_question_date and entry.get('question_date'):
            message = '(Current date: {}) {}'.format(entry['question_date'], message)
        payload = {
            'message': message,
            'end_user_id': qid2user[entry['question_id']],
            'config_id': config_id,
            'search_switch': search_switch,
        }
        start = time.time()
        body = fetch_one(session, url, headers, payload, args.timeout, args.max_retries)
        return body, round(time.time() - start, 2)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, entry): entry for entry in todo}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            entry = futures[fut]
            qid = entry['question_id']
            try:
                body, latency = fut.result()
            except Exception as err:
                failures.append((qid, str(err)))
                continue
            with out_lock:
                trials.append({
                    'trial_id': new_trial_id(qid),
                    'run_ts': run_ts,
                    'question_id': qid,
                    'question_type': entry.get('question_type'),
                    'end_user_id': qid2user[qid],
                    'question': entry['question'],
                    'question_date': entry.get('question_date'),
                    'answer': entry.get('answer'),
                    'hypothesis': None,
                    'latency_s': latency,
                    'read_response': body,
                    'reader': None,
                    'autoeval_label': None,
                })
                write_json_atomic(trials_file, trials)

    print('Done: {} fetched, {} failed, {} skipped.'.format(
        len(todo) - len(failures), len(failures), skipped))
    if failures:
        for qid, err in failures:
            print('  FAILED {}: {}'.format(qid, err))
        print('Rerun the same command to retry the failed questions.')
    print('Trials ({} total) saved to {}'.format(len(trials), trials_file))


if __name__ == '__main__':
    main()