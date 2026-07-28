# /// script
# requires-python = ">=3.9"
# ///
"""Download the LoCoMo dataset (locomo10.json) from the official repository.

Quick reproduction (src/locomo/reproduce.py) does NOT need this — it runs from
the released memories under results/locomo/. The dataset is only needed by the
full pipeline (src/locomo/run_predictions.py, score.py) and for inspecting the
source conversations.

Usage:
    uv run data/locomo/download_locomo.py            # download if missing
    uv run data/locomo/download_locomo.py --force    # re-download and overwrite

The file is validated (valid JSON, 10 conversation records) before it replaces
any local copy. License note: LoCoMo data is CC BY-NC 4.0 (© Snap Inc.) —
non-commercial use with attribution; this repo does not redistribute it.
"""

import argparse
import json
import os
import sys
import urllib.request

URL = 'https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json'
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, 'locomo10.json')


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
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)

    try:
        data = json.load(open(tmp))
    except json.JSONDecodeError as e:
        os.remove(tmp)
        sys.exit('downloaded file is not valid JSON: {}'.format(e))
    if not isinstance(data, list) or len(data) != 10:
        os.remove(tmp)
        sys.exit('unexpected dataset shape: expected a list of 10 conversations, '
                 'got {} with {} item(s)'.format(type(data).__name__,
                                                 len(data) if hasattr(data, '__len__') else '?'))

    os.replace(tmp, DEST)
    size_mb = os.path.getsize(DEST) / 1e6
    print('saved {} ({:.1f} MB, {} conversations, {} questions)'.format(
        DEST, size_mb, len(data), sum(len(s.get('qa', [])) for s in data)))


if __name__ == '__main__':
    main()
