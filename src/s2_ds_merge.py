import jsonlines
import gzip
import sys
import os
import pickle
from tqdm import tqdm
from pprint import pprint

if __name__ == "__main__":
    srcdir = sys.argv[1]
    dstpath = sys.argv[2]

    files = sorted([
        file
        for file in os.listdir(srcdir)
        if os.path.isfile(os.path.join(srcdir, file))
        and 's2-corpus' in file
        and '.jsonl' in file
    ])

    data = {}
    for file in files:
        with jsonlines.open(os.path.join(srcdir, file)) as reader:
            for obj in reader:
                id = obj['id']
                data[id] = obj

    with open(dstpath, 'wb') as f:
        pickle.dump(data, f)
