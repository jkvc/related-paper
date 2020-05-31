import torch
import torch.nn as nn
import torch.nn.functional as F
import dataloader
import sys
import train_utils
from transformers import AdamW
from triplet_loss import triplet_loss
from model_bert import *
from tqdm import tqdm, trange

BATCH_SIZE = 10
NUM_EPOCH = 2
EVAL_EVERY_N_BATCH = 2000
MARGIN = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def num_is_correct_eucl(ancs_encoding, poss_encoding, negs_encoding):
    distance_positive = (ancs_encoding - poss_encoding).pow(2).sum(1)
    distance_negative = (ancs_encoding - negs_encoding).pow(2).sum(1)
    # (B, )
    num_is_correct_eucl = (distance_positive < distance_negative).sum()
    return num_is_correct_eucl


if __name__ == "__main__":
    model_path = sys.argv[1]
    print(f'loading from [{model_path}]')
    model, existing_results = train_utils.load_model_save(model_path)

    optimizer = AdamW(model.parameters())

    train_loader, dev_loader, test_loader = dataloader.get_dataloaders(
        BATCH_SIZE)
    max_epoch = NUM_EPOCH * len(train_loader) // EVAL_EVERY_N_BATCH

    def train_epoch_fn(e):
        total_train_loss = 0
        total_num_correct_eucl = 0
        total_item = 0
        for i, data in enumerate(tqdm(train_loader, desc='train', leave=False)):
            if i > EVAL_EVERY_N_BATCH:
                break
            optimizer.zero_grad()

            ancs, poss, negs = data
            ancs = ancs.to(DEVICE)
            poss = poss.to(DEVICE)
            negs = negs.to(DEVICE)

            # (B, E)
            ancs_encoding = model(ancs)
            poss_encoding = model(poss)
            negs_encoding = model(negs)

            # train_loss = loss_fn(ancs, poss, negs)
            train_loss = triplet_loss(
                ancs_encoding, poss_encoding, negs_encoding, margin=MARGIN)
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()
            total_num_correct_eucl += num_is_correct_eucl(
                ancs_encoding, poss_encoding, negs_encoding
            ).item()
            total_item += len(ancs)

        results = {
            'train_loss': total_train_loss / total_item,
            'train_acc_eucl': total_num_correct_eucl / total_item,
        }
        return results

    def dev_epoch_fn(e):
        total_dev_loss = 0
        total_num_correct_eucl = 0
        total_item = 0
        for i, data in enumerate(tqdm(dev_loader, desc='dev', leave=False)):
            ancs, poss, negs = data
            ancs = ancs.to(DEVICE)
            poss = poss.to(DEVICE)
            negs = negs.to(DEVICE)

            # (B, E)
            ancs_encoding = model(ancs)
            poss_encoding = model(poss)
            negs_encoding = model(negs)

            dev_loss = triplet_loss(
                ancs_encoding, poss_encoding, negs_encoding, margin=MARGIN)
            total_dev_loss += dev_loss.item()
            total_num_correct_eucl += num_is_correct_eucl(
                ancs_encoding, poss_encoding, negs_encoding
            ).item()
            total_item += len(ancs)

        results = {
            'dev_loss': total_dev_loss / total_item,
            'dev_acc_eucl': total_num_correct_eucl / total_item,
        }
        return results

    results = train_utils.train_and_save(
        model,
        train_epoch_fn=train_epoch_fn,
        dev_epoch_fn=dev_epoch_fn,
        max_epoch=max_epoch,
        results=existing_results,
        save_model_path=model_path,
    )
