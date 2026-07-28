# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "openai>=1.35,<2",
#     "backoff>=2.2",
#     "tqdm>=4.66",
#     "regex",
#     "nltk",
#     "numpy",
# ]
# ///
"""Reproduce MemoryBear's LoCoMo results from the released memories.

Self-contained one-command reproduction (no MemoryBear service access needed):

    uv run src/locomo/reproduce.py

For every question in results/locomo/memorybear/memorybear_locomo_retrieved_memories.json.gz
(the per-question context MemoryBear retrieved — both speakers' memory stores
merged — released with this repo), this script:

  1. reader — sends the retrieved context + LoCoMo's per-category answer
     instructions to a reader LLM to produce a short answer (`hypothesis`).
     The prompt is built by prompts.build_messages, identical to
     run_predictions.py, so the reading protocol matches the published results
     (temperature=0, max_tokens=64, same as the original run).
  2. token-F1 — scores the hypothesis with LoCoMo's own
     eval_question_answering (vendored verbatim in task_eval/): category
     2/3/4 token-F1, category 1 multi-answer F1, category 5 = 1 only on a
     refusal ("no information available" / "not mentioned").
  3. LLM-judge — judges category 1-4 as CORRECT/WRONG with the same judge
     prompt as score.py (semantic match; list golds match on any one item,
     +/-14-day date tolerance); category 5 keeps the refusal rule, no LLM.

Output: results/locomo/reproduce_<reader_model>.json — one record per question
with question_id, category, question, golden_answer, hypothesis, f1 and the
judge label — written incrementally, so an interrupted run resumes where it
left off (already-judged questions are skipped; use --force to redo
everything). While running, every judged question prints a live line (verdict,
question → answer; suppress with --quiet);
both score tables (token-F1 and LLM-judge, per category and overall) are
printed at the end.

Config (.env at the repo root, or environment variables):
    LLM_API_KEY       required; your LLM provider's API key (reader + judge)
    LLM_BASE_URL      optional; the provider's OpenAI-compatible endpoint
    LLM_MODEL         optional; default model for both reader and judge
                      (--reader-model / --judge-model flags override it)
    (OPENAI_API_KEY / OPENAI_BASE_URL also work, as fallbacks)

Usage:
    uv run src/locomo/reproduce.py
    uv run src/locomo/reproduce.py --reader-model gpt-4o --judge-model gpt-4o
    uv run src/locomo/reproduce.py --workers 8
    uv run src/locomo/reproduce.py --force
"""

import argparse
import contextlib
import gzip
import io
import json
import os
import re
import sys
import threading
import time
import types
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import backoff
import openai
from openai import OpenAI
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))       # src/locomo/
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))      # repo root
DEFAULT_INPUT = os.path.join(REPO_ROOT, 'results', 'locomo', 'memorybear',
                             'memorybear_locomo_retrieved_memories.json.gz')

sys.path.insert(0, HERE)
from prompts import build_messages  # noqa: E402  (same reader prompt as run_predictions.py)

# LoCoMo's evaluation.py imports bert_score at module top but never calls it on
# the QA F1 path; stub it so we don't need transformers/torch (same shim as score.py).
_stub = types.ModuleType('bert_score')
_stub.score = lambda *a, **k: None
sys.modules.setdefault('bert_score', _stub)
from task_eval.evaluation import eval_question_answering  # noqa: E402  (vendored verbatim)

CATEGORY_NAMES = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain',
                  4: 'single-hop', 5: 'adversarial'}

REFUSAL = ('no information available', 'not mentioned')

# Verbatim from score.py — the judge protocol used for the published results.
JUDGE_SYSTEM = (
    "You are a grader for a question-answering benchmark. You are given a "
    "question, the reference (gold) answer, and a model's answer. Decide if the "
    "model's answer is CORRECT.\n"
    "- PARTIAL CREDIT: if the gold answer is a list of items (separated by "
    "commas, semicolons, or 'and'), the model answer is CORRECT if it matches AT "
    "LEAST ONE item. Matching 1 out of 2, or 2 out of 4, is CORRECT. Only mark "
    "WRONG if NONE of the gold items appear.\n"
    "- PARAPHRASES COUNT: the same concept in different words is CORRECT. Judge "
    "semantic meaning, not wording. For emotions/feelings questions, any emotion "
    "of the same polarity (positive/negative) about the same event is CORRECT "
    "('proud' matches 'fulfilled' or 'accomplished').\n"
    "- SAME REFERENT: if the model answer names or identifies the same specific "
    "entity, person, place, event, or work as the gold answer, it is CORRECT, "
    "even if described differently or with extra detail.\n"
    "- EXTRA DETAIL IS FINE: a longer answer containing the gold answer's key "
    "fact plus additional information is CORRECT. Never penalize verbosity or "
    "added specifics.\n"
    "- DATE TOLERANCE: dates within 14 days of each other are CORRECT. Durations "
    "within 50% of each other are CORRECT (e.g. '5 months' matches 'six "
    "months'). A relative expression (e.g. 'the week before 25 May 2023', "
    "'first week of April') matches any specific date or date range that "
    "overlaps that window. A specific date consistent with a vague reference "
    "('a few years before 2023' vs 'a few years before January 23, 2023') is "
    "CORRECT. Converting a relative year to the actual year ('last year' said "
    "in 2023 -> '2022') is CORRECT.\n"
    "- ONLY mark WRONG if the model answer contains ZERO correct items from the "
    "gold answer, contradicts it, or addresses a different topic.\n"
    "Think briefly first (one or two sentences): when either answer involves a "
    "date, a weekend, a week, or a duration, work out the concrete calendar "
    "date(s) or window each expression refers to before comparing them. "
    "Then output the final verdict on its own LAST line, exactly one word: "
    "CORRECT or WRONG."
)


def load_env_file():
    candidates = [
        os.path.join(os.getcwd(), '.env'),
        os.path.join(REPO_ROOT, '.env'),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            for line in open(cand):
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return cand
    return None


def load_questions(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def is_refusal(pred):
    p = (pred or '').lower()
    return any(r in p for r in REFUSAL)


def parse_verdict(text):
    # Verbatim from score.py: verdict on the last non-empty line, negatives first.
    t = (text or '').strip().lower()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    tail = lines[-1] if lines else ''
    if 'incorrect' in tail or 'wrong' in tail:
        return 0
    if 'correct' in tail:
        return 1
    if 'incorrect' in t or 'wrong' in t:
        return 0
    if 'correct' in t:
        return 1
    return 0


def compute_f1(record):
    """Token-F1 for one answered record via LoCoMo's eval_question_answering,
    unmodified (fed a single-item qa list; its own progress print silenced)."""
    qa = {'question': record['question'], 'answer': record['golden_answer'],
          'category': record['category'], 'evidence': [],
          'prediction': record['hypothesis']}
    with contextlib.redirect_stdout(io.StringIO()):
        ems, _lens, _recall = eval_question_answering([qa], 'prediction')
    return round(float(ems[0]), 3)


# Retry rate limits, network hiccups and 5xx; auth/config errors are NOT
# retried — they surface immediately in the per-question failure list.
@backoff.on_exception(backoff.expo,
                      (openai.RateLimitError, openai.APIConnectionError,
                       openai.InternalServerError),
                      max_tries=8)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def relpath(p):
    try:
        r = os.path.relpath(p, REPO_ROOT)
    except ValueError:
        return p
    return p if r.startswith('..') else r


_USE_COLOR = sys.stdout.isatty() and not os.environ.get('NO_COLOR')


def _mark(ok):
    if ok:
        return '\033[32m✓\033[0m' if _USE_COLOR else '✓'
    return '\033[31m✗\033[0m' if _USE_COLOR else '✗'


def _short(text, n):
    text = ' '.join(str(text or '').split())
    return text if len(text) <= n else text[:n - 1] + '…'


def print_run_header(args, questions, done, pending, base_url, out_file):
    convs = len({q['conv_id'] for q in questions})
    bar = '=' * 62
    print(bar)
    print('  MemoryBear benchmark reproduction — LoCoMo')
    print(bar)
    print('  input     : {}'.format(relpath(args.input_file)))
    print('  questions : {} ({} conversations)'.format(len(questions), convs))
    print('  reader    : {}'.format(args.reader_model))
    print('  judge     : {} (category 5 judged by refusal rule, no LLM)'.format(args.judge_model))
    print('  endpoint  : {}'.format(base_url or 'https://api.openai.com/v1 (default)'))
    print('  workers   : {}'.format(args.workers))
    print('  output    : {}'.format(relpath(out_file)))
    if done:
        print('  resume    : {} already done, {} remaining'.format(done, len(pending)))
    print('-' * 62)


def print_tables(records):
    done = [r for r in records.values() if r.get('label') is not None]
    if not done:
        return
    f1s, labels = defaultdict(list), defaultdict(list)
    for r in done:
        f1s[r['category']].append(r['f1'])
        labels[r['category']].append(r['label'])
    all_f1 = [v for vs in f1s.values() for v in vs]
    all_lab = [v for vs in labels.values() for v in vs]

    print()
    print('  Table 1: LoCoMo token-F1 (official)')
    print('  {:<6} {:<12} {:>6} {:>8}'.format('cat', 'category', '#', 'F1'))
    for cat in sorted(f1s):
        v = f1s[cat]
        print('  {:<6} {:<12} {:>6} {:>8.3f}'.format(
            cat, CATEGORY_NAMES.get(cat, '?'), len(v), sum(v) / len(v)))
    print('  {:<6} {:<12} {:>6} {:>8.3f}'.format(
        '', 'overall', len(all_f1), sum(all_f1) / len(all_f1)))

    print()
    print('  Table 2: LLM-as-judge accuracy')
    print('  {:<6} {:<12} {:>6} {:>8}'.format('cat', 'category', '#', 'acc'))
    for cat in sorted(labels):
        v = labels[cat]
        print('  {:<6} {:<12} {:>6} {:>7.1f}%'.format(
            cat, CATEGORY_NAMES.get(cat, '?'), len(v), 100 * sum(v) / len(v)))
    print('  {:<6} {:<12} {:>6} {:>7.1f}%'.format(
        '', 'overall', len(all_lab), 100 * sum(all_lab) / len(all_lab)))
    print()
    print('  Category legend: 1=multi-hop  2=temporal  3=open-domain  4=single-hop  5=adversarial')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file', nargs='?', default=DEFAULT_INPUT,
                        help='released memories json(.gz) (default: results/locomo/memorybear/memorybear_locomo_retrieved_memories.json.gz)')
    parser.add_argument('--reader-model', default=None,
                        help='model that turns the retrieved context into a short answer (default: LLM_MODEL from .env, else gpt-4o)')
    parser.add_argument('--judge-model', default=None,
                        help='LLM-judge model, same judge prompt as score.py (default: LLM_MODEL from .env, else gpt-4o)')
    parser.add_argument('--workers', type=int, default=4,
                        help='concurrent questions (default: 4)')
    parser.add_argument('--output', default=None,
                        help='output json (default: results/locomo/reproduce_<reader_model>.json)')
    parser.add_argument('--force', action='store_true',
                        help='redo all questions, ignoring previous output')
    parser.add_argument('--quiet', action='store_true',
                        help='suppress the live per-question judge lines')
    args = parser.parse_args()

    env_used = load_env_file()
    args.reader_model = args.reader_model or os.getenv('LLM_MODEL') or 'gpt-4o'
    args.judge_model = args.judge_model or os.getenv('LLM_MODEL') or 'gpt-4o'
    api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        sys.exit('Missing config: LLM_API_KEY. Set it in .env (loaded: {}) or the environment.'.format(
            env_used or 'no .env found'))
    base_url = os.getenv('LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL') or None

    if not os.path.isfile(args.input_file):
        sys.exit('Input file not found: {}'.format(args.input_file))
    questions = load_questions(args.input_file)

    out_file = args.output or os.path.join(
        REPO_ROOT, 'results', 'locomo',
        'reproduce_{}.json'.format(re.sub(r'[^A-Za-z0-9._-]+', '-', args.reader_model)))
    records = {}
    if os.path.isfile(out_file) and not args.force:
        records = {r['question_id']: r for r in json.load(open(out_file))}

    pending = [q for q in questions
               if (records.get(q['question_id']) or {}).get('label') is None]
    done = len(questions) - len(pending)
    print_run_header(args, questions, done, pending, base_url, out_file)
    if not pending:
        print('  Nothing to do — all {} questions already answered and judged.'.format(len(questions)))
        print_tables(records)
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    lock = threading.Lock()

    def save():
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        ordered = [records[q['question_id']] for q in questions if q['question_id'] in records]
        tmp = out_file + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_file)

    def process(q):
        qid = q['question_id']
        record = records.get(qid) or {
            'question_id': qid,
            'conv_id': q.get('conv_id'),
            'category': q['category'],
            'question_type': q.get('question_type'),
            'question': q['question'],
            'golden_answer': q.get('golden_answer'),
            'hypothesis': None,
            'f1': None,
            'label': None,
        }
        if not record.get('hypothesis'):
            messages = build_messages(
                {'category': q['category'], 'question': q['question']},
                q['retrieved_memories'], q['speaker_a'], q['speaker_b'])
            completion = chat_completions_with_backoff(
                client, model=args.reader_model, messages=messages,
                n=1, temperature=0, max_tokens=64)
            record['hypothesis'] = completion.choices[0].message.content.strip()
            record['reader_model'] = args.reader_model
            record['f1'] = compute_f1(record)
        if q['category'] == 5:
            # adversarial: correct == abstained. Rule-based, no LLM (same as score.py).
            record['label'] = 1 if is_refusal(record['hypothesis']) else 0
        else:
            user = ('Question: %s\nGold answer: %s\nModel answer: %s\nVerdict (CORRECT or WRONG):'
                    % (q['question'], str(q.get('golden_answer', '')), record['hypothesis']))
            completion = chat_completions_with_backoff(
                client, model=args.judge_model,
                messages=[{'role': 'system', 'content': JUDGE_SYSTEM},
                          {'role': 'user', 'content': user}],
                n=1, temperature=0, max_tokens=64)
            record['label'] = parse_verdict(completion.choices[0].message.content)
            record['judge_model'] = args.judge_model
        with lock:
            records[qid] = record
            save()
            if not args.quiet:
                done_n = sum(1 for r in records.values() if r.get('label') is not None)
                line = '  {} {:>4}/{} | {} → {}'.format(
                    _mark(record['label']), done_n, len(questions),
                    _short(record['question'], 56), _short(record['hypothesis'], 40))
                if not record['label']:
                    line += '  (gold: {})'.format(_short(record.get('golden_answer'), 32))
                tqdm.write(line)

    t0 = time.time()
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, q): q['question_id'] for q in pending}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc='  reader+judge', unit='q'):
            try:
                future.result()
            except Exception as e:
                failed.append((futures[future], e))

    elapsed = time.time() - t0
    print('-' * 62)
    print('  processed {}/{} questions in {}m{:02d}s -> {}'.format(
        len(pending) - len(failed), len(pending), int(elapsed // 60), int(elapsed % 60),
        relpath(out_file)))
    if failed:
        print('  {} question(s) failed; re-run the same command to retry them:'.format(len(failed)))
        for qid, e in failed[:10]:
            print('    {}: {}'.format(qid, e))
        if len(failed) > 10:
            print('    ... and {} more'.format(len(failed) - 10))
    print_tables(records)


if __name__ == '__main__':
    main()
