# /// script
# requires-python = ">=3.9"
# ///
"""Download the LongMemEval (cleaned) dataset from Hugging Face.

Quick reproduction (src/lme/reproduce.py) does NOT need this — it runs from the
released memories under results/lme/. The dataset is only needed by the full
pipeline (ingestion + src/lme/fetch_mem_hyp.py's LONGMEMEVAL_REF_FILE) and for
inspecting the source chat histories.

Downloads longmemeval_s_cleaned.json (~277 MB, 500 questions with ~115k-token
haystack chat histories — the file our published run used) from
https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned.

Usage:
    uv run data/lme/download_lme.py            # download if missing
    uv run data/lme/download_lme.py --force    # re-download and overwrite
"""

import argparse
import json
import os
import sys
import urllib.request

URL = ('https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/'
       'resolve/main/longmemeval_s_cleaned.json')
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, 'longmemeval_s_cleaned.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='overwrite an existing file')
    args = parser.parse_args()

    if os.path.exists(DEST) and not args.force:
        print('already present: {} (use --force to re-download)'.format(DEST))
        return

    print('downloading {} ...'.format(URL))
    tmp = DEST + '.tmp'
    with urllib.request.urlopen(URL) as resp, open(tmp, 'wb') as f:
        total = int(resp.headers.get('Content-Length') or 0)
        got = 0
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                sys.stdout.write('\r  {:5.1f}% of {:.0f} MB'.format(100 * got / total, total / 1e6))
                sys.stdout.flush()
        if total:
            print()

    try:
        data = json.load(open(tmp))
    except json.JSONDecodeError as e:
        os.remove(tmp)
        sys.exit('downloaded file is not valid JSON: {}'.format(e))
    if not isinstance(data, list) or len(data) != 500:
        os.remove(tmp)
        sys.exit('unexpected dataset shape: expected a list of 500 questions, '
                 'got {} with {} item(s)'.format(type(data).__name__,
                                                 len(data) if hasattr(data, '__len__') else '?'))

    os.replace(tmp, DEST)
    print('saved {} ({:.1f} MB, {} questions)'.format(DEST, os.path.getsize(DEST) / 1e6, len(data)))


if __name__ == '__main__':
    main()
