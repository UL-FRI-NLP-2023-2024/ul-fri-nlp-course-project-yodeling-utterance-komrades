import json
import re

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from torch.utils.data import DataLoader

from utils.multi_task_data_loader import DataLoader as MultiTaskDataLoader
from multi_task_run import config

tsdae_config = {
    # Training parameters.
    'epochs': 1,
    'weight_decay': 0.0,
    'scheduler': 'constantlr', # constantlr, warmupconstant, warmuplinear, warmupcosine
    'optimizer_params': {'lr': 0.0001},
    'batch_size': 8,

    # Data preprocessing parameters.
    'num_sentences': 20000,
}

if __name__ == '__main__':
    data_loader = MultiTaskDataLoader(
        '../data/SentiNews_sentiment/SentiNews_train.json', 
        '../data/SentiNews_sentiment/SentiNews_test.json',
        chunk=True
    )

    train_data, train_labels, train_sentiment, test_data, test_labels, test_sentiment = data_loader.load_data()

    torch.cuda.empty_cache()

    print('Training TSDAE model...', flush=True)
    print('Preprocessing data...', flush=True)

    sentences = []
    num_sentences = 0
    # Training data is chunked, so a list of lists.
    for row in train_data:
        for chunk in row:
            sentences.append(chunk)
            num_sentences += 1
        
        if num_sentences > tsdae_config['num_sentences']:
            break

    print('Number of sentences:', len(sentences), flush=True)

    # This dataset class already contains the noise functionality.
    train_data = DenoisingAutoEncoderDataset(sentences)
    loader = DataLoader(train_data, batch_size=tsdae_config['batch_size'], shuffle=True, drop_last=True)
    model = SentenceTransformer('sentence-transformers/LaBSE')
    model = model.to('cuda')
    loss = DenoisingAutoEncoderLoss(model=model, tie_encoder_decoder=True)
    loss = loss.to('cuda')

    print('Training...', flush=True)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=tsdae_config['epochs'],
        weight_decay=tsdae_config['weight_decay'],
        scheduler=tsdae_config['scheduler'] if tsdae_config['scheduler'] else None,
        optimizer_params=tsdae_config['optimizer_params'],
        show_progress_bar=False
    )

    print('Saving model...', flush=True)

    model.save(f'../models/sentence_transformers/tsdae_{config["saved_model_name"]}')
    with open(f'../models/sentence_transformers/tsdae_{config["saved_model_name"]}_config.json', 'w') as file:
        json.dump(tsdae_config, file)
