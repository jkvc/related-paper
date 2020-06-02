import pickle
from tqdm import tqdm
import transformers
import sys

# bert_tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
#         'distilbert-base-uncased')
bert_tokenizer = transformers.BertTokenizerFast.from_pretrained(
    'bert-base-uncased')

if __name__ == "__main__":
    corpus_path = sys.argv[1]
    bert_token_save_path = sys.argv[2]

    with open(corpus_path, 'rb') as f:
        data = pickle.load(f)

    bert_data = {}
    for id in tqdm(data):
        obj = data[id]
        abstract = obj['paperAbstract']
        tokens = bert_tokenizer.tokenize(abstract)
        encoded = [bert_tokenizer.cls_token_id] + \
            bert_tokenizer.convert_tokens_to_ids(tokens)
        encoded = encoded[:512]
        bert_data[id] = encoded

    with open(bert_token_save_path, 'wb') as f:
        pickle.dump(bert_data, f)
