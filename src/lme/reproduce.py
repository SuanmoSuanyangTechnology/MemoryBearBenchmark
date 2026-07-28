# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "openai>=1.35,<2",
#     "backoff>=2.2",
#     "tqdm>=4.66",
# ]
# ///
"""Reproduce MemoryBear's LongMemEval results from the released memories.

Self-contained one-command reproduction (no MemoryBear service access needed):

    uv run src/lme/reproduce.py

For every question in results/lme/memorybear/memorybear_lme_retrieved_memories.json (the
per-question memories MemoryBear retrieved, released with this repo), this
script:

  1. reader — sends question + question-type guidance + the retrieved memory
     to a reader LLM to produce a concise answer (`hypothesis`). The memory is
     built from the structured `intermediate_outputs` items and the prompt is
     identical to generate_hyp_interm.py, so the reading protocol matches the
     published results.
  2. judge — checks the hypothesis against the golden answer with the original
     LongMemEval LLM-judge prompts (verbatim from evaluate_qa.py, including the
     unanswerable `_abs` template), n=1 / temperature=0 / max_tokens=10,
     'yes' in the reply => correct.

Output: results/lme/reproduce_<reader_model>.json — one record per question with
question_id, question_type, question, golden_answer, hypothesis and the judge
label — written incrementally, so an interrupted run resumes where it left off
(already-judged questions are skipped; use --force to redo everything).
While running, every judged question prints a live line (verdict,
question → answer; suppress with --quiet); overall and
per-question-type accuracy are printed at the end.

Config (.env at the repo root, or environment variables):
    LLM_API_KEY       required; your LLM provider's API key (reader + judge)
    LLM_BASE_URL      optional; the provider's OpenAI-compatible endpoint
    LLM_MODEL         optional; default model for both reader and judge
                      (--reader-model / --judge-model flags override it)
    (OPENAI_API_KEY / OPENAI_BASE_URL also work, as fallbacks)

Usage:
    uv run src/lme/reproduce.py
    uv run src/lme/reproduce.py --reader-model gpt-4o --judge-model gpt-4o
    uv run src/lme/reproduce.py --workers 8
    uv run src/lme/reproduce.py --force
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import backoff
import openai
from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DEFAULT_INPUT = os.path.join(REPO_ROOT, 'results', 'lme', 'memorybear',
                             'memorybear_lme_retrieved_memories.json')


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


# Per question_type guidance that nudges the reader toward the form of answer the
# judge (get_anscheck_prompt below) expects for that category. Verbatim from
# generate_hyp_interm.py so the reading protocol matches the published results.
TYPE_GUIDANCE = {
    'temporal-reasoning': (
        "This question is about timing/order, and you ARE given the CURRENT DATE "
        "(today) in this prompt. Use it as the anchor: first resolve EVERY relative time "
        "expression — in both the question and the memory — to a concrete calendar "
        "date (e.g. 'X days/weeks/months ago', 'last Tuesday', 'two months ago', "
        "'this past weekend', or a memory note that says 'today'/'yesterday'). Then "
        "order the events and compute the asked-for date, count, or duration "
        "explicitly. If a number of days/weeks/months is requested, state the number. "
        "Because today's date is given, NEVER refuse on the grounds that the current "
        "date is unknown — always compute against it. Only say it is unanswerable if "
        "the underlying event or its date is genuinely absent from the memory."
    ),
    'knowledge-update': (
        "This fact changed over time; the memory holds several dated values for it. "
        "Order them by their dates/timestamps — not by wording like 'recent', and not by "
        "which value appears most often — then answer what is asked: the latest value by "
        "default; the older value if the question asks for the 'previous'/former one; or "
        "BOTH values plus the direction of change if it asks before-vs-now or whether it "
        "went up/down. For a count/total, add the base plus any later additions/removals "
        "instead of reporting the base. For 'how long', give a duration (e.g. '3 months'), "
        "not a start date. If the question anchors the timeframe to an event that is missing "
        "from the memory (e.g. 'before I bought the gravel bike'), still answer the "
        "underlying fact rather than refusing."
    ),
    'single-session-preference': (
        "This is a recommendation/advice request, NOT a fact lookup. The memory holds "
        "the user's PREFERENCES, not the answer. You MUST give a concrete, helpful "
        "response using your own knowledge — never say it is unanswerable or that the "
        "information is not present. Personalize it to the user's remembered "
        "preferences and details (brands, past successes, constraints) and weave those "
        "specifics in. Transfer a preference even if it was expressed about a different "
        "topic, place, or time than the question (e.g. a hotel preference formed in one "
        "city still applies to another). Honor negative preferences: avoid what the "
        "user said they dislike or want to move beyond."
    ),
    'single-session-user': (
        "Answer directly from what the user stated in the memory."
    ),
    'multi-session': (
        "The answer may be spread across several sessions; synthesize across the "
        "memory entries to form one answer."
    ),
    'single-session-assistant': (
        "Recall the specific detail the question asks for; it is in the memory, "
        "possibly inside a larger table, list, or story — match the question's "
        "qualifier to the right item and return just that value. Ignore who said "
        "it: the content may read as the user's words or be in second person "
        "rather than yours; do not refuse on that basis."
    ),
    'abstention': (
        "Check carefully whether the memory actually contains the specific "
        "information the question asks for. If it does not, clearly state that the "
        "information is not available in the memory instead of guessing or inferring "
        "an answer."
    ),
}


def build_reader_prompt(question_type, question, memory, question_id='', question_date=None):
    # Detect abstention the same way the judge does (_abs suffix).
    is_abstention = '_abs' in (question_id or '') or question_type == 'abstention'
    guidance = (TYPE_GUIDANCE['abstention'] if is_abstention
                else TYPE_GUIDANCE.get(question_type, "Answer directly using the information in the memory."))
    # Give the reader "today" so it can resolve relative dates and compute durations.
    # Scoped to non-abstention temporal-reasoning only: abstention (_abs) questions are
    # left untouched so an injected date can't tempt them away from abstaining.
    show_date = (question_type == 'temporal-reasoning' and not is_abstention and question_date)
    date_block = ("CURRENT DATE (when the question was asked): {}\n\n".format(question_date)
                  if show_date else "")
    return (
        "You are the answer-generation component (the \"reader\") of a long-term "
        "memory question-answering system. You are given a USER QUESTION, its "
        "QUESTION TYPE, and a block of MEMORY that was retrieved about this user. "
        "Produce a concise, direct answer to the question.\n\n"
        "Rules:\n"
        "- By default, ground your answer in the MEMORY: do not bring in outside facts "
        "or invent details — UNLESS the question-type guidance below tells you to.\n"
        "- By default, if the MEMORY lacks the information the question asks for, reply "
        "that it cannot be answered (the information is not present / unanswerable) and "
        "do NOT guess — UNLESS the question-type guidance below tells you to still respond.\n"
        "- Be concise: give the answer itself, not a summary of the memory, and no "
        "preamble such as \"Based on the memory...\".\n"
        "- Guidance for this question type ({qtype}) — this OVERRIDES the defaults above "
        "when they conflict: {guidance}\n\n"
        "{date_block}"
        "QUESTION: {question}\n\n"
        "MEMORY:\n{memory}\n\n"
        "ANSWER:"
    ).format(qtype=question_type, guidance=guidance, date_block=date_block,
             question=question, memory=memory)


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    # Verbatim from evaluate_qa.py (the original LongMemEval judge prompts).
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


def build_memory(retrieved_memories):
    """Build the reader's MEMORY block from the released retrieved memories.

    Same construction as generate_hyp_interm.py: each intermediate_outputs item
    contributes its `content` block, and ExtractedEntity items additionally get
    their `data.description` — timestamped detail (e.g. "[2023-04-10T17:15:00Z]
    ...") that is absent from `content`.
    """
    items = (retrieved_memories or {}).get('intermediate_outputs')
    if not items:
        return None
    blocks = []
    for it in items:
        content = (it.get('content') or '').strip()
        desc = (((it.get('data') or {}).get('description')) or '').strip()
        if not content and not desc:
            continue
        block = content
        if desc:
            # '；' separates the dated statements; one per line reads better.
            block = (block + '\nDETAILS:\n' + desc.replace('；', '\n')).strip()
        blocks.append(block)
    return '\n\n'.join(blocks) if blocks else None


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
    bar = '=' * 62
    print(bar)
    print('  MemoryBear benchmark reproduction — LongMemEval')
    print(bar)
    print('  input     : {}'.format(relpath(args.input_file)))
    print('  questions : {}'.format(len(questions)))
    print('  reader    : {}'.format(args.reader_model))
    print('  judge     : {} (original LongMemEval judge prompts)'.format(args.judge_model))
    print('  endpoint  : {}'.format(base_url or 'https://api.openai.com/v1 (default)'))
    print('  workers   : {}'.format(args.workers))
    print('  output    : {}'.format(relpath(out_file)))
    if done:
        print('  resume    : {} already done, {} remaining'.format(done, len(pending)))
    print('-' * 62)


def print_accuracy(records):
    labeled = [r for r in records.values() if r.get('label') is not None]
    if not labeled:
        return
    qtype2labels = {}
    for r in labeled:
        qtype2labels.setdefault(r.get('question_type'), []).append(1 if r['label'] else 0)
    all_labels = [v for vs in qtype2labels.values() for v in vs]
    print()
    print('  LLM-judge accuracy (original LongMemEval protocol)')
    print('  {:<28} {:>6} {:>8}'.format('question type', '#', 'acc'))
    for k in sorted(qtype2labels):
        v = qtype2labels[k]
        print('  {:<28} {:>6} {:>7.1f}%'.format(k, len(v), 100 * sum(v) / len(v)))
    print('  {:<28} {:>6} {:>7.1f}%'.format(
        'overall', len(all_labels), 100 * sum(all_labels) / len(all_labels)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file', nargs='?', default=DEFAULT_INPUT,
                        help='released memories json (default: results/lme/memorybear/memorybear_lme_retrieved_memories.json)')
    parser.add_argument('--reader-model', default=None,
                        help='model that turns memory into an answer (default: LLM_MODEL from .env, else gpt-4o)')
    parser.add_argument('--judge-model', default=None,
                        help='LLM-judge model, original LongMemEval protocol (default: LLM_MODEL from .env, else gpt-4o)')
    parser.add_argument('--workers', type=int, default=4,
                        help='concurrent questions (default: 4)')
    parser.add_argument('--output', default=None,
                        help='output json (default: results/lme/reproduce_<reader_model>.json)')
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
    questions = json.load(open(args.input_file))

    out_file = args.output or os.path.join(
        REPO_ROOT, 'results', 'lme',
        'reproduce_{}.json'.format(re.sub(r'[^A-Za-z0-9._-]+', '-', args.reader_model)))
    records = {}
    if os.path.isfile(out_file) and not args.force:
        records = {r['question_id']: r for r in json.load(open(out_file))}

    no_memory = [q['question_id'] for q in questions if not build_memory(q.get('retrieved_memories'))]
    pending = [q for q in questions
               if build_memory(q.get('retrieved_memories'))
               and (records.get(q['question_id']) or {}).get('label') is None]
    done = len(questions) - len(pending) - len(no_memory)
    print_run_header(args, questions, done, pending, base_url, out_file)
    if no_memory:
        print('  Warning: {} question(s) have no retrieved memory to read from; skipping.'.format(len(no_memory)))
    if not pending:
        print('  Nothing to do — all {} questions already answered and judged.'.format(len(questions)))
        print_accuracy(records)
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
            'question_type': q.get('question_type'),
            'question': q['question'],
            'golden_answer': q.get('golden_answer'),
            'hypothesis': None,
            'label': None,
        }
        if not record.get('hypothesis'):
            prompt = build_reader_prompt(
                q.get('question_type'), q['question'], build_memory(q['retrieved_memories']),
                qid, q.get('question_date'))
            completion = chat_completions_with_backoff(
                client, model=args.reader_model,
                messages=[{"role": "user", "content": prompt}],
                n=1, temperature=0, max_tokens=512)
            record['hypothesis'] = completion.choices[0].message.content.strip()
            record['reader_model'] = args.reader_model
        prompt = get_anscheck_prompt(
            q.get('question_type'), q['question'], q.get('golden_answer'),
            record['hypothesis'], abstention='_abs' in qid)
        completion = chat_completions_with_backoff(
            client, model=args.judge_model,
            messages=[{"role": "user", "content": prompt}],
            n=1, temperature=0, max_tokens=10)
        record['label'] = 'yes' in completion.choices[0].message.content.strip().lower()
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
                    line += '  (gold: {})'.format(_short(record.get('golden_answer'), 36))
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
    print_accuracy(records)


if __name__ == '__main__':
    main()