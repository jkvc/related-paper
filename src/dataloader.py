import torch
import torch.nn.functional as F
import os
import pickle
import transformers
from pprint import pprint
from tqdm import tqdm

TRIPLET_PATH = '../data/s2_arxiv_triplets.pkl'
BERT_ENCODING_PATH = '../data/s2_corpus_bertencoded.pkl'

MAX_BERT_LENGTH = 512
BERT_PAD_TOKEN = transformers.BertTokenizerFast.from_pretrained(
    'bert-base-uncased').pad_token_id


def get_dataloaders(batch_size):
    with open(TRIPLET_PATH, 'rb') as f:
        triplets = pickle.load(f)
    with open(BERT_ENCODING_PATH, 'rb') as f:
        bert_encoding_dict = pickle.load(f)

    dataloaders = (
        torch.utils.data.DataLoader(
            TripletDataset(
                triplets[split],
                bert_encoding_dict
            ),
            batch_size=batch_size,
            num_workers=1,
            shuffle=True
        )
        for split in ['train', 'dev', 'test']
    )
    return dataloaders


def pad(tensor):
    return F.pad(
        tensor,
        pad=(0, MAX_BERT_LENGTH - tensor.shape[0]),
        mode='constant',
        value=BERT_PAD_TOKEN
    )


class TripletDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        id_triplets,
        bert_encoding_dict
    ):
        self.id_triplets = id_triplets
        self.bert_encoding_dict = bert_encoding_dict

    def __len__(self):
        return len(self.id_triplets)

    def __getitem__(self, idx):
        src_id, pos_id, neg_id = self.id_triplets[idx]
        src_encoding, pos_encoding, neg_encoding = (
            torch.tensor(self.bert_encoding_dict[src_id]),
            torch.tensor(self.bert_encoding_dict[pos_id]),
            torch.tensor(self.bert_encoding_dict[neg_id]),
        )
        src_encoding, pos_encoding, neg_encoding = (
            pad(src_encoding),
            pad(pos_encoding),
            pad(neg_encoding),
        )

        return (src_encoding, pos_encoding, neg_encoding)


if __name__ == "__main__":

    train_loader, dev_loader, test_loader = get_dataloaders(10)

    for i, data in enumerate(tqdm(train_loader)):
        srcs, poss, negs = data
        # print(srcs.shape)
        # break
