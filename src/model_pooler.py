import torch
import sys
import transformers
import pickle
import train_utils

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

BERT_DIM = 768


class Pooler(torch.nn.Module):
    def __init__(self, out_dim):
        super(Pooler, self).__init__()
        self.out_dim = out_dim
        layers = []
        feat_in = 768 * 3  # cls, max, avg
        self.fc = torch.nn.Linear(feat_in, out_dim)

    def forward(self, x):
        return self.fc(x)


out_dim = {
    'small': BERT_DIM,
    'medium': BERT_DIM * 2,
    'large': BERT_DIM * 3
}

if __name__ == "__main__":
    outpath_prefix = sys.argv[1]
    for config_name in out_dim:
        model = Pooler(out_dim[config_name])
        train_utils.init_model_save(model, f'{outpath_prefix}_{config_name}')
