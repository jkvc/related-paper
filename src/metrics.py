import torch
import torch.nn as nn
import torch.nn.functional as F


def triplet_loss(anchor, positive, negative, margin):
    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.sum()


def num_is_correct_eucl(ancs_encoding, poss_encoding, negs_encoding):
    distance_positive = (ancs_encoding - poss_encoding).pow(2).sum(1)
    distance_negative = (ancs_encoding - negs_encoding).pow(2).sum(1)
    # (B, )
    num_is_correct_eucl = (distance_positive < distance_negative).sum()
    return num_is_correct_eucl
