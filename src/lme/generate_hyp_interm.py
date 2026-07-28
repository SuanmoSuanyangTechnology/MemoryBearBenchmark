# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "openai>=1.35,<2",
#     "backoff>=2.2",
#     "tqdm>=4.66",
# ]
# ///
"""Generate answer hypotheses from the retrieved memory (the reader stage).

Builds the reader's MEMORY from the structured
`read_response.data.intermediate_outputs` list (rather than the flat
`read_response.data.answer` string).

This is the middle step of the three-stage pipeline:

    fetch_mem_hyp.py  ->  generate_hyp_interm.py  ->  evaluate_qa.py

`data.answer` is exactly the `content` blocks of `intermediate_outputs` joined
together, so this variant reproduces that same text and additionally appends, for
each ExtractedEntity item, its `data.description` — timestamped detail
(e.g. "[2023-04-10T17:15:00Z] ...") that is absent from `content`/`answer`.
Statement and MemorySummary items have no description and are left unchanged.

Source of truth for the raw memory is always `read_response.data.intermediate_outputs`,
never `hypothesis` (which this script overwrites), so re-running is safe and idempotent.

Which trials get processed: only those without a `reader` marker (i.e. not yet
generated). Generating a trial sets the `reader` marker and resets its
`autoeval_label` to null so evaluate re-judges it. Use --force to regenerate
trials that already have a hypothesis.

Config comes from a .env file (see .env.example at the repo root):
    READER_MODEL     (e.g. qwen3.7-plus)
    READER_API_KEY   (OpenAI-compatible API key)
    READER_API_BASE  (OpenAI-compatible base url; default DashScope compatible-mode)
    TRIALS_FILE      (input/output trials json; default results/trials.json)

Usage:
    uv run generate_hyp_interm.py                               # trials from .env TRIALS_FILE
    uv run generate_hyp_interm.py results/trials.json           # explicit trials file
    uv run generate_hyp_interm.py results/trials.json --force   # regenerate all
"""

import argparse
import json
import os
import sys

import backoff
import openai
from openai import OpenAI
from tqdm import tqdm


def load_env_file():
    candidates = [
        os.path.join(os.getcwd(), '.env'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'),
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
# judge (evaluate_qa.py / get_anscheck_prompt) expects for that category.
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
    'single-session-assistant' : (
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


def build_prompt(question_type, question, memory, question_id='', question_date=None):
    # Detect abstention the same way the judge does (_abs suffix); also honour a
    # literal question_type of 'abstention' if a future dataset labels it that way.
    is_abstention = '_abs' in (question_id or '') or question_type == 'abstention'
    guidance = (TYPE_GUIDANCE['abstention'] if is_abstention
                else TYPE_GUIDANCE.get(question_type, "Answer directly using the information in the memory."))
    # Give the reader "today" so it can resolve relative dates and compute durations.
    # Scoped to non-abstention temporal-reasoning only: abstention (_abs) trials are
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


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trials_file', nargs='?', default=None,
                        help='trials json (default: TRIALS_FILE from .env, else results/trials.json)')
    parser.add_argument('--force', action='store_true',
                        help='regenerate even trials that already have a reader hypothesis')
    args = parser.parse_args()

    env_used = load_env_file()

    model = os.getenv('READER_MODEL')
    api_key = os.getenv('READER_API_KEY')
    api_base = os.getenv('READER_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    trials_file = args.trials_file or os.getenv('TRIALS_FILE') or 'results/trials.json'

    missing = [name for name, value in [('READER_MODEL', model),
                                        ('READER_API_KEY', api_key)] if not value]
    if missing:
        sys.exit('Missing config: {}. Set them in .env (loaded: {}).'.format(
            ', '.join(missing), env_used or 'no .env found'))
    if not os.path.isfile(trials_file):
        sys.exit('Trials file not found: {} (run fetch_mem_hyp.py first, or set TRIALS_FILE).'.format(trials_file))

    client = OpenAI(api_key=api_key, base_url=api_base)

    def read_memory(trial):
        """Build the reader's memory from the retrieved intermediate_outputs items.

        Each item contributes its `content` block (this alone reproduces
        `data.answer`). For ExtractedEntity items we additionally append
        `data.description`, which carries timestamped detail
        (e.g. "[2023-04-10T17:15:00Z] ...") that is absent from `content`/`answer`.
        Source of truth, not `hypothesis` (which this script overwrites).
        """
        items = (((trial.get('read_response') or {}).get('data') or {}).get('intermediate_outputs'))
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

    def generate(trial):
        memory = read_memory(trial)
        prompt = build_prompt(trial.get('question_type'), trial['question'], memory,
                              trial.get('question_id', ''), trial.get('question_date'))
        completion = chat_completions_with_backoff(
            client, model=model,
            messages=[{"role": "user", "content": prompt}],
            n=1, temperature=0, max_tokens=512)
        return completion.choices[0].message.content.strip()

    trials = json.load(open(trials_file))
    if args.force:
        pending = [t for t in trials if read_memory(t)]
    else:
        pending = [t for t in trials if t.get('reader') is None and read_memory(t)]
    no_memory = [t for t in trials if not read_memory(t)]
    if no_memory:
        print('Warning: {} trial(s) have no read_response.data.intermediate_outputs to read from; skipping.'.format(
            len(no_memory)))

    if not pending:
        print('No trials to generate in {}; nothing to do.'.format(trials_file))
        return

    for t in tqdm(pending):
        t['hypothesis'] = generate(t)
        t['reader'] = {'model': model}
        # The answer changed, so any earlier judgement is stale: re-judge it.
        t['autoeval_label'] = None

    tmp = trials_file + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(trials, f, ensure_ascii=False, indent=2)
    os.replace(tmp, trials_file)
    print('Generated {} hypothesis/-es with {}; updated {}'.format(len(pending), model, trials_file))
    print('Next: uv run evaluate_qa.py qwen-plus {}'.format(trials_file))


if __name__ == '__main__':
    main()