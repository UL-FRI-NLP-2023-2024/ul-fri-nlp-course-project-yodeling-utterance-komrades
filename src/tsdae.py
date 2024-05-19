import nltk
nltk.download('punkt')
import torch
import re
import json
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from torch.utils.data import DataLoader

from utils.multi_task_data_loader import DataLoader as MultiTaskDataLoader
from multi_task_run import config

tsdae_config = {
    'epochs': 1,
    'weight_decay': 0,
    'scheduler': 'constantlr', # constantlr, warmupconstant, warmuplinear, warmupcosine
    'optimizer_params': {'lr': 3e-5},
}

if __name__ == '__main__':
    data_loader = MultiTaskDataLoader(
        '../data/SentiNews_sentiment/SentiNews_train.json', 
        '../data/SentiNews_sentiment/SentiNews_test.json',
        chunk=False
    )

    train_data, train_labels, train_sentiment, test_data, test_labels, test_sentiment = data_loader.load_data()

    torch.cuda.empty_cache()

    sentence_splitter = re.compile(r'\.\s?\n?')
    sentences = []
    num_sentences = 0
    for row in train_data:
        new_sentences = sentence_splitter.split(row)
        new_sentences = [sentence for sentence in new_sentences if len(sentence) > 10]

        sentences.extend(new_sentences)
        num_sentences += len(new_sentences)

        # Sentence transformers recommends to limit the number of sentences to 10-100k.
        if num_sentences > 100_000:
            break

    # This dataset class already contains the noise functionality.
    train_data = DenoisingAutoEncoderDataset(sentences)
    loader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)

    model = models.Transformer('sentence-transformers/LaBSE')
    # A pooling layer is necessary to move the output of 512 token vectors to a single sentence vector.
    pooling = models.Pooling(model.get_word_embedding_dimension(), 'cls')

    model = SentenceTransformer(modules=[model, pooling])

    loss = DenoisingAutoEncoderLoss(model=model, tie_encoder_decoder=True)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=tsdae_config['epochs'],
        weight_decay=tsdae_config['weight_decay'],
        scheduler=tsdae_config['scheduler'],
        optimizer_params=tsdae_config['optimizer_params'],
        show_progress_bar=False
    )

    model.save(f'../models/sentence_transformers/tsdae_{config["saved_model_name"]}')
    with open(f'../models/sentence_transformers/tsdae_{config["saved_model_name"]}_config.json', 'w') as file:
        json.dump(tsdae_config, file)



