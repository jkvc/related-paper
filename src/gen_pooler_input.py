import torch
import torch.nn as nn
import torch.nn.functional as F
import dataloader
import sys
import train_utils
import transformers
from transformers import AdamW
from triplet_loss import triplet_loss
from model_bert import *
from tqdm import tqdm, trange
from model_pooler import *

BERT_ENCODING_PATH = '../data/s2_corpus_bertencoded.pkl'
POOLER_INPUT_PATH = '../data/s2_corpus_poolerinput.pkl'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY = 100


def stack_pooler_input(hidden):
    # (B,L,E)
    cls_encoding = hidden[:, 0, :]
    avg_encoding = hidden.mean(1)
    max_encoding, _ = hidden.max(1)
    pooler_input = torch.cat(
        [cls_encoding, avg_encoding, max_encoding], dim=1)
    return pooler_input


if __name__ == "__main__":
    bert = transformers.DistilBertModel.from_pretrained(
        'distilbert-base-uncased').to(DEVICE)

    with open(BERT_ENCODING_PATH, 'rb') as f:
        bert_encoding_dict = pickle.load(f)

    pooler_inputs = {}
    for i, id in enumerate(tqdm(bert_encoding_dict)):
        with torch.no_grad():
            bert_encoding = bert_encoding_dict[id]
            bert_hidden, = bert(torch.tensor([bert_encoding]).to(DEVICE))
            pooler_input = stack_pooler_input(bert_hidden).squeeze(0)
            pooler_inputs[id] = pooler_input.cpu().numpy()

        if (i+1) % SAVE_EVERY == 0:
            with open(POOLER_INPUT_PATH, 'wb') as f:
                pickle.dump(pooler_inputs, f)
