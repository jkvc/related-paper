import pickle
from tqdm import tqdm
import transformers
import sys

if __name__ == "__main__":
    srcpath = sys.argv[1]
    dstpath = sys.argv[2]

    with open(srcpath, 'rb') as f:
        data = pickle.load(f)

    bert_tokenizer = transformers.BertTokenizerFast.from_pretrained(
        'bert-base-uncased')

    bert_data = {}
    for id in tqdm(data):
        obj = data[id]
        abstract = obj['paperAbstract']
        tokens = bert_tokenizer.tokenize(abstract)
        encoded = [bert_tokenizer.cls_token_id] + \
            bert_tokenizer.convert_tokens_to_ids(tokens)
        encoded = encoded[:512]
        bert_data[id] = encoded

    with open(dstpath, 'wb') as f:
        pickle.dump(bert_data, f)
