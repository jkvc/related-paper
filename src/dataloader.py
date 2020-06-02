import torch
import torch.nn.functional as F
import os
import pickle
import transformers
import s2_ds_sample_triplets
from pprint import pprint
from tqdm import tqdm

MAX_BERT_LENGTH = 512

# BERT_ENCODING_PATH = '../data/s2_corpus_bertencoded_bertbase.pkl'
# BERT_PAD_TOKEN = transformers.BertTokenizerFast.from_pretrained(
#     'bert-base-uncased').pad_token_id

BERT_ENCODING_PATH = '../data/s2_corpus_bertencoded_distilbert.pkl'
BERT_PAD_TOKEN = transformers.DistilBertTokenizerFast.from_pretrained(
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
            shuffle=(split == 'train')  # only shuffle trian set
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


TRIPLET_PATH = '../data/s2_triplets.pkl'
CITATION_PATH = '../data/s2_corpus_citation.pkl'


def get_pooling_dataloaders(batch_size, pooler_input_path, sampled_train_set=False):
    with open(TRIPLET_PATH, 'rb') as f:
        triplets = pickle.load(f)
    with open(pooler_input_path, 'rb') as f:
        pooler_input_dict = pickle.load(f)
    with open(CITATION_PATH, 'rb') as f:
        citation_dict = pickle.load(f)

    if not sampled_train_set:
        train_loader = torch.utils.data.DataLoader(
            TripletPoolingDataset(
                triplets['train'],
                pooler_input_dict
            ),
            batch_size=batch_size,
            num_workers=1,
            shuffle=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            TripletPoolingDatasetSampled(
                triplets['train'],
                pooler_input_dict,
                citation_dict
            ),
            batch_size=batch_size,
            num_workers=1,
            shuffle=True
        )

    dev_loader, test_loader = (
        torch.utils.data.DataLoader(
            TripletPoolingDataset(
                triplets[split],
                pooler_input_dict
            ),
            batch_size=batch_size,
            num_workers=1,
            shuffle=False
        )
        for split in ['dev', 'test']
    )
    return train_loader, dev_loader, test_loader


class TripletPoolingDatasetSampled(torch.utils.data.Dataset):
    def __init__(
        self,
        id_triplets,
        pooler_input_dict,
        citation_dict
    ):
        self.id_triplets = id_triplets
        self.pooler_input_dict = pooler_input_dict
        self.citation_dict = citation_dict

        self.all_ids = list(citation_dict.keys())

    def __len__(self):
        return len(self.twins)

    def __getitem__(self, idx):
        src_id, pos_id, _ = self.id_triplets[idx]
        neg_id = s2_ds_sample_triplets.get_negative_sample(
            self.all_ids, self.citation_dict[src_id]
        )
        src_encoding, pos_encoding, neg_encoding = (
            torch.tensor(self.pooler_input_dict[src_id]),
            torch.tensor(self.pooler_input_dict[pos_id]),
            torch.tensor(self.pooler_input_dict[neg_id]),
        )

        return (src_encoding, pos_encoding, neg_encoding)


class TripletPoolingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        id_triplets,
        pooler_input_dict
    ):
        self.id_triplets = id_triplets
        self.pooler_input_dict = pooler_input_dict

    def __len__(self):
        return len(self.id_triplets)

    def __getitem__(self, idx):
        src_id, pos_id, neg_id = self.id_triplets[idx]
        src_encoding, pos_encoding, neg_encoding = (
            torch.tensor(self.pooler_input_dict[src_id]),
            torch.tensor(self.pooler_input_dict[pos_id]),
            torch.tensor(self.pooler_input_dict[neg_id]),
        )

        return (src_encoding, pos_encoding, neg_encoding)


if __name__ == "__main__":

    train_loader, dev_loader, test_loader = get_pooling_dataloaders(10)

    for i, data in enumerate(tqdm(train_loader)):
        srcs, poss, negs = data
        print(srcs.shape)
        # break
