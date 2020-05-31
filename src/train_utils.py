import os
import json
import numpy as np
from tqdm import trange, tqdm
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_model_save(model, save_model_path):
    assert (not os.path.exists(save_model_path)
            ), f'{save_model_path} already exists'
    os.mkdir(save_model_path)

    newest_model_path = os.path.join(save_model_path, 'newest.pth')
    best_model_path = os.path.join(save_model_path, 'best.pth')
    results_path = os.path.join(save_model_path, 'results.json')
    model_specs_path = os.path.join(save_model_path, 'model_specs.txt')

    results = {
        'train_loss': [],
        'dev_loss': [],
        'test_loss': [],
    }

    torch.save(model, newest_model_path)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    with open(model_specs_path, 'w') as f:
        f.write(model.__repr__())
        f.write('\n')


def load_model_save(save_model_path, device=DEVICE):
    newest_model_path = os.path.join(save_model_path, 'newest.pth')
    results_path = os.path.join(save_model_path, 'results.json')

    model = torch.load(newest_model_path, map_location=device)

    results = None
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    return model, results


def train_and_save(
    model,
    train_epoch_fn,
    max_epoch,
    results,
    save_model_path,
    dev_epoch_fn=None,
    device=DEVICE
):
    model = model.to(device=device)

    newest_model_path = os.path.join(save_model_path, 'newest.pth')
    best_model_path = os.path.join(save_model_path, 'best.pth')
    results_path = os.path.join(save_model_path, 'results.json')
    print(f'Will save model checkpoints to [{save_model_path}]')

    epoch_start = len(results['train_loss'])
    print(f'starting epoch={epoch_start}, end epoch={max_epoch}')

    best_dev_loss = min(results['dev_loss'], default=float('inf'))
    print(f'Starting train, best_dev_loss=[{best_dev_loss}]')

    for e in trange(epoch_start, max_epoch, desc='EPOCH'):
        model.train()
        train_result = train_epoch_fn(e)
        merge_results(results, train_result)
        save_results(results, results_path)
        torch.save(model, newest_model_path)

        if dev_epoch_fn is not None:
            model.eval()
            with torch.no_grad():
                dev_result = dev_epoch_fn(e)
            merge_results(results, dev_result)
            save_results(results, results_path)
            new_dev_loss = dev_result['dev_loss']
            if new_dev_loss < best_dev_loss:
                print('dev loss improved')
                best_dev_loss = new_dev_loss
                torch.save(model, best_model_path)

    return results


def save_results(results, path):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def merge_results(results, new_results):
    for k in new_results:
        if k not in results:
            results[k] = []
        results[k].append(new_results[k])
