import torch
import sys
import transformers
import pickle
import train_utils

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

BERT_DIM = 768


class Pooler(torch.nn.Module):
    def __init__(self, sizes):
        super(Pooler, self).__init__()
        layers = []
        feat_in = 768 * 3  # cls, max, avg
        for feat_out in sizes:
            layers.append(
                torch.nn.Linear(
                    feat_in, feat_out, bias=True
                )
            )
            layers.append(torch.nn.LeakyReLU())
            feat_in = feat_out

        layers.append(
            torch.nn.Linear(
                feat_in, BERT_DIM, bias=True
            )
        )
        layers.append(torch.nn.Tanh())
        self.fc_module = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_module(x)


layer_configs = {
    'small': [],
    'large': [BERT_DIM * 2],
}

if __name__ == "__main__":
    outpath_prefix = sys.argv[1]
    for config_name in layer_configs:
        model = Pooler(layer_configs[config_name])
        train_utils.init_model_save(model, f'{outpath_prefix}_{config_name}')
