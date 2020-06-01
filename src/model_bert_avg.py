import torch
import sys
import transformers
import pickle
import train_utils

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class BertAvgModel(torch.nn.Module):
    def __init__(
        self,
        bert_module,
    ):
        super(BertAvgModel, self).__init__()
        self.bert_module = bert_module

    def forward(self, x):
        hidden, = self.bert_module(x)  # (B,L,E)
        encoding = hidden.mean(1)
        return encoding


if __name__ == "__main__":

    # with open('../data/metadata_bert_tokens.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # bert_tokens = data[list(data.keys())[0]]
    # bert_tokens = [101,  2057,  6235,  1037,  2047,  9896,  1010,  1996,  1002,  1006,
    #                1047,  1010,  1032,  3449,  2140,  1007,  1002,  1011, 21877, 11362,
    #                2208,  2007,  6087,  1010,  1998,  2224,  2009,  6855,  1037, 23191,
    #                1997,  1996,  2155,  1997,  1002,  1006,  1047,  1010,  1032,  3449,
    #                2140,  1007,  1002,  1011, 20288, 19287,  1998,  9896,  2594,  7300,
    #                2000,  1037,  2155,  1997,  3471,  7175,  3392, 22511,  2015,  1997,
    #                19287,  1012,  2569, 12107,  1997, 20288, 19287,  3711,  1999, 11841,
    #                3012,  3399,  1998,  2031,  2363,  3445,  3086,  1999,  3522,  2086,
    #                1012,  1999,  3327,  1010,  2256,  6910, 28962,  2236,  4697,  1998,
    #                12919,  1996,  3025,  3463,  1997,  3389,  1998,  2358,  2890,  2378,
    #                2226,  1998,  2507,  1037,  2047,  6947,  1997,  1996, 10722,  4674,
    #                1011, 10594,  1011,  3766, 23191,  1997, 19679, 28775,  3723,  1012,
    #                2057,  2036,  2556,  1037,  2047, 22511,  2008,  8292, 28228, 14213,
    #                12403,  2869,  3012,  2241,  2006,  1996,  1002,  1006,  1047,  1010,
    #                1032,  3449,  2140,  1007,  1002,  1011, 21877, 11362,  2208,  2007,
    #                6087,  1012,  2256,  2147,  2036, 14451,  2015,  7264,  2090, 21877,
    #                11362,  2208, 13792,  1998,  3025, 20288, 10629, 13792,  2011, 11721,
    #                18912,  1010, 11721, 18912,  1998,  2225, 18689,  2078,  1998, 28895,
    #                2239,  1012]
    # bert_tokens = torch.tensor([bert_tokens]).to(DEVICE)
    # print(bert_tokens)
    # bert_module = transformers.DistilBertModel.from_pretrained(
    #     'distilbert-base-uncased').to(DEVICE)
    # model = BertAvgModel(bert_module)
    # print(model)

    # bert_hidden, = bert_module(bert_tokens)
    # print(bert_tokens.shape)
    # print(bert_hidden.shape)
    # print(bert_hidden)

    bert_module = transformers.DistilBertModel.from_pretrained(
        'distilbert-base-uncased').to(DEVICE)
    model = BertAvgModel(bert_module)
    train_utils.init_model_save(model, sys.argv[1])
