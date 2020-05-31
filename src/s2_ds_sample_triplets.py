import os
import pickle
import sys
import random
from tqdm import tqdm

random.seed(420)
TRAINPROP, DEVPROP, TESTPROP = 0.95, 0.025, 0.025


def get_samples(src_ids, all_ids, citation_graph):
    samples = []
    for src_id in tqdm(src_ids):
        pos_ids = citation_graph[src_id]
        for pos_id in pos_ids:
            neg_id = get_negative_sample(all_ids, pos_ids)
            samples.append((src_id, pos_id, neg_id))
    return samples


def get_negative_sample(all_ids, pos_ids):
    while True:
        neg_id = random.choice(all_ids)
        if neg_id not in pos_ids:
            return neg_id


if __name__ == "__main__":
    srcpath = sys.argv[1]
    dstpath = sys.argv[2]

    with open(srcpath, 'rb') as f:
        citation_data = pickle.load(f)

    ids = list(citation_data.keys())
    random.shuffle(ids)

    num_article = len(ids)
    num_train = int(num_article * TRAINPROP)
    num_dev = int(num_article * DEVPROP)
    num_test = int(num_article * TESTPROP)

    train_src_ids = ids[:num_train]
    dev_src_ids = ids[num_train:num_train+num_dev]
    test_src_ids = ids[num_train+num_dev:]
    print(
        'train dev test split:',
        len(train_src_ids), len(dev_src_ids), len(test_src_ids)
    )

    train_samples = get_samples(train_src_ids, ids, citation_data)
    dev_samples = get_samples(dev_src_ids, ids, citation_data)
    test_samples = get_samples(test_src_ids, ids, citation_data)
    print(
        'train dev test samples:',
        len(train_samples), len(dev_samples), len(test_samples)
    )

    with open(dstpath, 'wb') as f:
        pickle.dump({
            'train': train_samples,
            'dev': dev_samples,
            'test': test_samples,
        }, f)
