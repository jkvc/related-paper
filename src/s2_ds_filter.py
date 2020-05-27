import jsonlines
import gzip
import sys
import os
from tqdm import tqdm
from pprint import pprint
from langdetect import detect
import json
import concurrent.futures

NUM_WORKER = 8
KEEP_KEYS = [
    'id',
    'inCitations', 'outCitations',
    'year',
    'paperAbstract',
    'title',
]


def is_cs(obj):
    return 'Computer Science' in obj['fieldsOfStudy']


def is_arxiv(obj):
    return obj['journalName'] == 'ArXiv'


def has_citation(obj):
    return len(obj['outCitations']) > 0 or len(obj['inCitations']) > 0


def has_abstract(obj):
    return len(obj['paperAbstract']) > 0


def is_english(obj):
    return detect(obj['paperAbstract']) == 'en'


def get_narrowed_obj(obj):
    return {
        k: obj[k]
        for k in KEEP_KEYS
    }


def run_filter(srcpath, dstpath):

    print(f'begin [{srcpath}]')
    sys.stdout.flush()

    writer = open(dstpath, 'w')

    with open(srcpath) as f:
        while True:
            line = f.readline()
            if not line:
                break

            try:
                obj = json.loads(line)
                if is_cs(obj) and is_arxiv(obj) and has_citation(obj) and has_abstract(obj) and is_english(obj):
                    writer.write(json.dumps(get_narrowed_obj(obj)))
                    writer.write('\n')
            except KeyboardInterrupt:
                return
            except:
                continue

    writer.close()

    print(f'done [{srcpath}]')
    sys.stdout.flush()


if __name__ == "__main__":
    srcdir = sys.argv[1]
    dstdir = sys.argv[2]

    files = sorted([
        file
        for file in os.listdir(srcdir)
        if os.path.isfile(os.path.join(srcdir, file))
        and 's2-corpus' in file
    ])

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKER) as executor:
        for file in files:
            executor.submit(
                run_filter,
                os.path.join(srcdir, file),
                os.path.join(dstdir, file+'.jsonl')
            )
