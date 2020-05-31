import torch
import torch.nn as nn
import torch.nn.functional as F
import dataloader
import sys
import train_utils
from transformers import AdamW
from triplet_loss import TripletLoss
from model_bert import *
from tqdm import tqdm, trange

BATCH_SIZE = 20
NUM_EPOCH = 5

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
    model, existing_results = torch.load(model_path, map_location=DEVICE)

    optimizer = AdamW(model.parameters())
    loss_fn = TripletLoss(margin=1)

    train_loader, dev_loader, test_loader = dataloader.get_dataloaders(
        BATCH_SIZE)

    def train_epoch_fn(e):
        total_train_loss = 0
        total_num_correct_eucl = 0
        total_item = 0
        for i, data in enumerate(tqdm(train_loader, desc='train', leave=False)):
            optimizer.zero_grad()

            ancs, poss, negs = data
            ancs = ancs.to(DEVICE)
            poss = poss.to(DEVICE)
            negs = negs.to(DEVICE)

            # (B, E)
            ancs_encoding = model(ancs)
            poss_encoding = model(poss)
            negs_encoding = model(negs)

            train_loss = loss_fn(ancs, poss, negs)
            total_train_loss += train_loss.item() * len(ancs)
            total_num_correct_eucl += num_is_correct_eucl(
                ancs_encoding, poss_encoding, negs_encoding
            )
            total_item += len(ancs)

            train_loss.backward()
            optimizer.step()

        results = {
            'train_loss': total_train_loss / total_item,
            'train_acc_eucl': total_num_correct_eucl / total_item,
        }
        return results

    def dev_epoch_fn(e):
        total_dev_loss = 0
        total_num_correct_eucl = 0
        total_item = 0
        for i, data in enumerate(tqdm(dev_loader, desc='train', leave=False)):
            ancs, poss, negs = data
            ancs = ancs.to(DEVICE)
            poss = poss.to(DEVICE)
            negs = negs.to(DEVICE)

            # (B, E)
            ancs_encoding = model(ancs)
            poss_encoding = model(poss)
            negs_encoding = model(negs)

            dev_loss = loss_fn(ancs, poss, negs)
            total_dev_loss += dev_loss.item() * len(ancs)
            total_num_correct_eucl += num_is_correct_eucl(
                ancs_encoding, poss_encoding, negs_encoding
            )
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
        max_epoch=NUM_EPOCH,
        results=existing_results,
        save_model_path=model_path,
    )
