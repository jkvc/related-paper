from model_infersent import InferSent
import torch

V = 1
MODEL_PATH = 'infersent_data/encoder/infersent%s.pkl' % V
params_model = {
    'bsize': 64,
    'word_emb_dim': 300,
    'enc_lstm_dim': 2048,
    'pool_type': 'max',
    'dpout_model': 0.0,
    'version': V
}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = 'infersent_data/fastText/crawl-300d-2M.vec'
infersent.set_w2v_path(W2V_PATH)
