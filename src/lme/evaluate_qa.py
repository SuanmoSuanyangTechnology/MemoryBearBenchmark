# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "openai>=1.35,<2",
#     "backoff>=2.2",
#     "numpy>=1.26",
#     "tqdm>=4.66",
# ]
# ///
import os
import sys
import json
from tqdm import tqdm
import backoff
import openai
from openai import OpenAI
import numpy as np


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
            return


load_env_file()


model_zoo = {
    'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
    'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
    'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
    'qwen-plus': ('qwen-plus', 'dashscope'),
    'qwen3.7-plus': ('qwen3.7-plus', 'dashscope')
}


@backoff.on_exception(backoff.expo, (openai.RateLimitError,
                                    openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def get_anscheck_prompt(task, question, answer, response, abstention=False):
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


if __name__ == '__main__':
    argv = [a for a in sys.argv[1:] if a != '--legacy']
    legacy = '--legacy' in sys.argv[1:]
    if len(argv) < 1 or len(argv) > 2:
        print('Usage: uv run evaluate_qa.py metric_model [trials_file] [--legacy]')
        print('trials_file defaults to TRIALS_FILE from .env (else results/trials.json).')
        print('--legacy also exports an original-style per-question eval-results jsonl.')
        exit()

    metric_model_short = argv[0]
    trials_file = argv[1] if len(argv) > 1 else (os.getenv('TRIALS_FILE') or 'results/trials.json')
    if not os.path.isfile(trials_file):
        print('Trials file not found: {} (run fetch_mem_hyp.py first, or set TRIALS_FILE).'.format(trials_file))
        exit()

    if metric_model_short not in model_zoo:
        print('Requested metric model is not supported:', metric_model_short)
        exit()
    metric_model, metric_model_source = model_zoo[metric_model_short]
    if metric_model_source == 'openai':
        openai.organization = os.getenv('OPENAI_ORGANIZATION')
        metric_api_key = os.getenv('OPENAI_API_KEY')
        metric_api_base = None
    elif metric_model_source == 'dashscope':
        metric_api_key = os.getenv('DASHSCOPE_API_KEY')
        metric_api_base = os.getenv('DASHSCOPE_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        if not metric_api_key:
            print('DASHSCOPE_API_KEY is not set. Add it to .env or export it.')
            exit()
    else:
        metric_api_key = "EMPTY"
        metric_api_base = "http://localhost:8001/v1"

    metric_client = OpenAI(
        api_key=metric_api_key,
        base_url=metric_api_base,
    )

    def judge(qtype, question, answer, hypothesis, qid):
        prompt = get_anscheck_prompt(qtype, question, answer, hypothesis, abstention='_abs' in qid)
        completion = chat_completions_with_backoff(
            metric_client, model=metric_model,
            messages=[{"role": "user", "content": prompt}],
            n=1, temperature=0, max_tokens=10)
        return 'yes' in completion.choices[0].message.content.strip().lower()

    # Judge only the new (un-labeled) trials that already have a generated
    # hypothesis; already-judged trials are left as-is, so re-running is
    # incremental and the full history is preserved.
    trials = json.load(open(trials_file))
    pending = [t for t in trials if t.get('autoeval_label') is None and t.get('hypothesis')]
    not_ready = [t for t in trials if t.get('autoeval_label') is None and not t.get('hypothesis')]
    if not_ready:
        print('Note: {} trial(s) have no hypothesis yet; run generate_hyp_interm.py first. Skipping them.'.format(
            len(not_ready)))
    for t in tqdm(pending):
        label = judge(t.get('question_type'), t['question'], t.get('answer'),
                      t['hypothesis'], t['question_id'])
        t['autoeval_label'] = {'model': metric_model, 'label': label}
    if pending:
        tmp = trials_file + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(trials, f, ensure_ascii=False, indent=2)
        os.replace(tmp, trials_file)
        print('Judged {} new trial(s); updated {}'.format(len(pending), trials_file))
    else:
        print('No un-judged trials in {}; nothing to do.'.format(trials_file))

    # Benchmark accuracy over the latest labeled trial per question (repeats deduped).
    latest = {}
    for t in trials:
        if t.get('autoeval_label') is not None:
            latest[t['question_id']] = t
    qtype2acc = {}
    for t in latest.values():
        qtype2acc.setdefault(t.get('question_type'), []).append(1 if t['autoeval_label']['label'] else 0)
    all_labels = [v for vs in qtype2acc.values() for v in vs]
    if all_labels:
        print('Accuracy (latest per question, {} questions): {}'.format(
            len(all_labels), round(np.mean(all_labels).item(), 4)))
        for k, v in qtype2acc.items():
            print('\t{}: {} ({})'.format(k, round(np.mean(v), 4), len(v)))

    # --legacy: export an original-style per-question eval-results jsonl (latest trial each).
    if legacy:
        result_file = os.path.join(os.path.dirname(trials_file) or '.',
                                   'eval-results-{}.jsonl'.format(metric_model_short))
        with open(result_file, 'w') as out_f:
            for t in latest.values():
                print(json.dumps({
                    'question_id': t['question_id'],
                    'question_type': t.get('question_type'),
                    'hypothesis': t['hypothesis'],
                    'autoeval_label': t['autoeval_label'],
                }, ensure_ascii=False), file=out_f)
        print('Legacy eval-results saved to', result_file)
