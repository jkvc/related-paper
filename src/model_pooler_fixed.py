import torch
import sys
import transformers
import pickle
import train_utils

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

BERT_DIM = 768


class PoolerFixed(torch.nn.Module):
    def __init__(self):
        super(PoolerFixed, self).__init__()
        feat_in = 768 * 3  # cls, max, avg
        self.fc = torch.nn.Linear(feat_in, BERT_DIM)

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    outpath = sys.argv[1]
    model = PoolerFixed()
    train_utils.init_model_save(model, outpath)
