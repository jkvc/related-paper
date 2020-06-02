from model_infersent import InferSent
import torch
import pickle
import sys
from tqdm import tqdm, trange
import nltk
nltk.download('punkt')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

V = int(sys.argv[1])
corpus_path = sys.argv[2]
infersent_encoding_save_path = sys.argv[3]

if V == 1:
    print('using glove')
    MODEL_PATH = 'infersent_data/encoder/infersent1.pkl'
    W2V_PATH = 'infersent_data/GloVe/glove.840B.300d.txt'
elif V == 2:
    print('using fasttext')
    MODEL_PATH = 'infersent_data/encoder/infersent2.pkl'
    W2V_PATH = 'infersent_data/fastText/crawl-300d-2M.vec'
else:
    raise ValueError()

if __name__ == "__main__":

    params_model = {
        'bsize': 64,
        'word_emb_dim': 300,
        'enc_lstm_dim': 2048,
        'pool_type': 'max',
        'dpout_model': 0.0,
        'version': V
    }
    infersent = InferSent(params_model).to(DEVICE)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    infersent.set_w2v_path(W2V_PATH)

    with open(corpus_path, 'rb') as f:
        data = pickle.load(f)

    ids = [k for k in data]
    sentences = [data[id]['paperAbstract'] for id in ids]

    infersent.build_vocab(sentences, tokenize=True)

    embeddings = infersent.encode(sentences, tokenize=True, bsize=64)

    split_len = len(ids) // 2
    infersent_encoded = {
        ids[idx]: embeddings[idx]
        for idx in trange(len(ids[:split_len]), desc='build_dict')
    }
    with open(infersent_encoding_save_path+'.1', 'wb') as f:
        pickle.dump(infersent_encoded, f)
    infersent_encoded = {
        ids[idx]: embeddings[idx]
        for idx in trange(len(ids[split_len:]), desc='build_dict')
    }
    with open(infersent_encoding_save_path+'.2', 'wb') as f:
        pickle.dump(infersent_encoded, f)
