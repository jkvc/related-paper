import pickle
from tqdm import tqdm
import transformers
import sys

if __name__ == "__main__":
    srcpath = sys.argv[1]
    dstpath = sys.argv[2]

    with open(srcpath, 'rb') as f:
        data = pickle.load(f)

    citation_data = {}
    for id in tqdm(data):
        obj = data[id]
        citations = set()
        for cid in obj['inCitations']:
            if cid in data:
                citations.add(cid)
        for cid in obj['outCitations']:
            if cid in data:
                citations.add(cid)
        citation_data[id] = list(citations)

    with open(dstpath, 'wb') as f:
        pickle.dump(citation_data, f)
