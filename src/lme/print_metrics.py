# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers", "huggingface_hub", "rouge-score", "nltk", "numpy"]
# ///
"""Compute memorybear's evaluation metrics from <lme_results>.json and generate memorybear_lme_metrics.json.

The output has three levels:
  - overall     : all 500 questions
  - by_category : 6 groups by question_type
  - by_question : per question (key = question_id)

Metric definitions:
  - llm_judge_score : mean of autoeval_label.label (True=1); judged once, so llm_judge_std=0
  - lexical         : hypothesis (prediction) vs answer (golden)
        f1        — SQuAD-style: lowercase / strip punctuation / drop articles, then whitespace-tokenize; multiset token-overlap F1
        rouge*_f  — fmeasure from rouge_score (use_stemmer=True)
        bleu1..4  — nltk sentence_bleu, word_tokenize (lowercased) + method1 smoothing, cumulative n-grams
        meteor    — nltk meteor_score, word_tokenize (lowercased)
  - context_tokens  : count read_response.data.answer with the Qwen tokenizer (Qwen/Qwen3-8B, same family as count_tokens.py)
  - duration        : mean / p50 / p95 of latency_s*1000

Run: uv run src/lme/print_metrics.py
"""

import json
import re
import string
from collections import Counter, defaultdict

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

RESULTS = "<path>"
OUT = "<path>"
TOKENIZER = "Qwen/Qwen3-8B"  # The whole Qwen3 family shares the same BPE vocab; proxy for the qwen3.7-plus reader
CATEGORIES = ['single-session-user', 'single-session-preference', 'single-session-assistant',
              'multi-session', 'temporal-reasoning', 'knowledge-update']

for pkg in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4'):
    nltk.download(pkg, quiet=True)

_enc = AutoTokenizer.from_pretrained(TOKENIZER)
_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
_smooth = SmoothingFunction().method1
_BLEU_W = {1: (1, 0, 0, 0), 2: (.5, .5, 0, 0), 3: (1/3, 1/3, 1/3, 0), 4: (.25, .25, .25, .25)}
_ARTICLES = re.compile(r'\b(a|an|the)\b')


def _squad_tokens(text: str) -> list[str]:
    text = text.lower()
    text = ''.join(ch if ch not in string.punctuation else ' ' for ch in text)
    text = _ARTICLES.sub(' ', text)
    return text.split()


def token_f1(pred: str, ref: str) -> float:
    p, r = _squad_tokens(pred), _squad_tokens(ref)
    ns = sum((Counter(p) & Counter(r)).values())
    if ns == 0 or not p or not r:
        return 0.0
    prec, rec = ns / len(p), ns / len(r)
    return 2 * prec * rec / (prec + rec)


def lexical_metrics(pred: str, ref: str) -> dict:
    rs = _rouge.score(ref, pred)  # (target, prediction)
    ref_tok, pred_tok = word_tokenize(ref.lower()), word_tokenize(pred.lower())
    out = {
        'f1': token_f1(pred, ref),
        'rouge1_f': rs['rouge1'].fmeasure,
        'rouge2_f': rs['rouge2'].fmeasure,
        'rougeL_f': rs['rougeL'].fmeasure,
    }
    for n in (1, 2, 3, 4):
        if pred_tok:
            out[f'bleu{n}'] = sentence_bleu([ref_tok], pred_tok, weights=_BLEU_W[n],
                                            smoothing_function=_smooth)
        else:
            out[f'bleu{n}'] = 0.0
    out['meteor'] = meteor_score([ref_tok], pred_tok) if pred_tok else 0.0
    return out


def context_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _s(x) -> str:
    return str(x) if x is not None else ''


def _agg_lexical(rows: list[dict]) -> dict:
    keys = ['f1', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor']
    return {k: float(np.mean([r['lexical'][k] for r in rows])) for k in keys}


def _latency(rows: list[dict]) -> dict:
    vals = [r['latency_ms'] for r in rows]
    return {
        'mean': float(np.mean(vals)),
        'p50': float(np.percentile(vals, 50)),
        'p95': float(np.percentile(vals, 95)),
    }


def _block(rows: list[dict]) -> dict:
    return {
        'num_questions': len(rows),
        'accuracy': float(np.mean([r['correct'] for r in rows])),
        'avg_context_tokens': float(np.mean([r['context_tokens'] for r in rows])),
        'latency_ms': _latency(rows),
        'lexical': _agg_lexical(rows),
    }


def main() -> None:
    with open(RESULTS, encoding='utf-8') as f:
        records = json.load(f)

    rows = []
    for rec in records:
        answer = ((rec.get('read_response') or {}).get('data') or {}).get('answer') or ''
        rows.append({
            'question_id': rec['question_id'],
            'category': rec['question_type'],
            'correct': bool((rec.get('autoeval_label') or {}).get('label')),
            'latency_ms': round(rec['latency_s'] * 1000, 3),
            'context_tokens': context_tokens(answer),
            'lexical': lexical_metrics(_s(rec.get('hypothesis')), _s(rec.get('answer'))),
        })

    report = {
        'system': 'memorybear',
        'dataset': 'longmemeval',
        'num_questions': len(rows),
        'overall': _block(rows),
        'by_category': {},
        'by_question': {},
    }
    for cat in CATEGORIES:
        crows = [r for r in rows if r['category'] == cat]
        if crows:
            report['by_category'][cat] = _block(crows)
    for r in rows:
        report['by_question'][r['question_id']] = {
            'correct': r['correct'],
            'context_tokens': r['context_tokens'],
            'latency_ms': r['latency_ms'],
            'lexical': r['lexical'],
        }

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    o = report['overall']
    print(f"\nwrote {OUT}  (num_questions={report['num_questions']})")
    print(f"accuracy        : {o['accuracy']:.4f}")
    print(f"avg_context_tok : {o['avg_context_tokens']:.1f}")
    print(f"latency_ms      : {o['latency_ms']['mean']:.1f} "
          f"(p50 {o['latency_ms']['p50']:.1f} / p95 {o['latency_ms']['p95']:.1f})")
    print("lexical         : " + "  ".join(f"{k}={v:.4f}" for k, v in o['lexical'].items()))
    print("\nby category:")
    for cat, blk in report['by_category'].items():
        print(f"  {cat:28s} n={blk['num_questions']:>3}  acc={blk['accuracy']:.4f}  "
              f"ctx_tok={blk['avg_context_tokens']:.0f}  f1={blk['lexical']['f1']:.4f}  "
              f"rougeL={blk['lexical']['rougeL_f']:.4f}  meteor={blk['lexical']['meteor']:.4f}")


if __name__ == '__main__':
    main()