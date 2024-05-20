import os
import re
import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample, losses

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

from utils.multi_task_data_loader import DataLoader as MultiTaskDataLoader
from multi_task_run import config

gpl_config = {
    'epochs': 1, 
}

if __name__ == '__main__':
    data_loader = MultiTaskDataLoader(
        '../data/SentiNews_sentiment/SentiNews_train.json', 
        '../data/SentiNews_sentiment/SentiNews_test.json',
        chunk=False
    )

    train_data, train_labels, train_sentiment, test_data, test_labels, test_sentiment = data_loader.load_data()

    sentence_splitter = re.compile(r'\.\s?\n?')
    sentences = []
    num_sentences = 0
    for row in train_data:
        new_sentences = sentence_splitter.split(row)
        new_sentences = [sentence for sentence in new_sentences if len(sentence) > 10]

        sentences.extend(new_sentences)
        num_sentences += len(new_sentences)

        # Sentence transformers recommends to limit the number of sentences to 10-100k.
        if num_sentences > 50000:
            break

    target_model = 'sentence-transformers/LaBSE'
    generator_name = 'google/mt5-base'
    #generator_name = 't5-base' #'t5-base-multilingual-translation'
    retriever_names = ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2']
    cross_encoder_name = 'sentence-transformers/LaBSE' # 'bert-base-multilingual-cased'

    print('Sentences generated')
    print('Generating triplets...')

    # Check if the data file already exists.
    if not os.path.exists('../data/triplets.tsv'):

        # Step 1: Query generation.   
        tokenizer = AutoTokenizer.from_pretrained(generator_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(generator_name).cuda()

        # Generate (query, passage) pairs.
        pairs = []
        counter = 0
        for sentence in sentences:

            inputs = tokenizer(sentence, return_tensors='pt')
            inputs = {key: val.cuda() for key, val in inputs.items()}
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=256,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1
            ).cpu()

            query = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pairs.append((query, sentence))

            if counter % 100 == 0:
                print(f'Generating query {counter}...')
                print(f'Query: {query}')
                print(f'Passage: {sentence}')
            counter += 1

        print('1: Query generation complete.')

        # Step 2: Negative mining.
        passage_batch = []
        id_batch = []
        batch_size = 64

        model = SentenceTransformer(target_model).cuda()

        embeddings = []
        for i, (query, passage) in enumerate(pairs):
            if passage not in passage_batch:
                passage_batch.append(passage)
                id_batch.append(i)

            if len(passage_batch) == batch_size or i == len(pairs) - 1:
                embeddings.extend(model.encode(passage_batch))
                passage_batch = []
                id_batch = []

        embeddings = np.array(embeddings)

        print('2: Negative mining complete.')

        # Step 3: Pseudo-labeling.
        batch_size = 100
        triplets = []

        for i in tqdm(range(0, len(pairs), batch_size)):
            i_end = min(i + batch_size, len(pairs))
            query_batch = [pair[0] for pair in pairs[i:i_end]]
            positive_batch = [pair[1] for pair in pairs[i:i_end]]

            # Create query embeddings
            query_embeddings = model.encode(
                query_batch, 
                convert_to_tensor=True, 
                show_progress_bar=False
            ).cpu().numpy()

            similarities = cosine_similarity(query_embeddings, embeddings)

            for query, positive, similarity in zip(query_batch, positive_batch, similarities):
                top_k_indices = np.argsort(similarity)[-10:]

                # Shuffle the indices to ensure randomness
                random.shuffle(top_k_indices)

                for index in top_k_indices:
                    negative = pairs[index][1]

                    # Check if the negative is not the same as the positive
                    if negative != positive:
                        triplets.append((query, positive, negative))
                        break 

        print('3: Pseudo-labeling complete.')

        with open('../data/triplets.tsv', 'w', encoding='utf-8') as file:
            for triplet in triplets:
                file.write('\t'.join(map(str, triplet)))

        model = CrossEncoder(cross_encoder_name)

        label_lines = []

        for line in triplets:
            q, p, n = line

            # Predict (Q, P+) nad (Q, P-) scores
            p_score = model.predict([(q, p)])
            n_score = model.predict([(q, n)])

            margin = p_score - n_score

            label_lines.append(
                f'{q}\t{p}\t{n}\t{str(margin[0])}'
            )

        print('4: Cross-encoder complete.')

        with open('../data/triplet_margins.tsv', 'w', encoding='utf-8') as file:
            file.write('\n'.join(label_lines))
    else:
        with open('../data/triplets.tsv', 'r', encoding='utf-8') as file:
            triplets = [line.strip().split('\t') for line in file.readlines()]

        with open('../data/triplet_margins.tsv', 'r', encoding='utf-8') as file:
            label_lines = [line.strip().split('\t') for line in file.readlines()]

    training_data = []
    for line in label_lines:
        if (type(line) == str):
            line = line.strip().split('\t')
        q, p, n, margin = line
        training_data.append(InputExample(texts=[q, p, n], label=float(margin)))

    print(f"""
        Query: {training_data[0].texts[0]}
        Positive: {training_data[0].texts[1]}
        Negative: {training_data[0].texts[2]}
        Margin: {training_data[0].label}
    """)
    
    print('Training data generated.')

    torch.cuda.empty_cache()

    batch_size = 32

    loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True
    )

    model = SentenceTransformer(target_model).cuda()
    loss = losses.MarginMSELoss(model).cuda()

    print('Training model...')

    warmup_steps = int(len(loader) * gpl_config['epochs'] * 0.1)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=gpl_config['epochs'],
        warmup_steps=warmup_steps,
        output_path=f'../models/sentence_transformers/gpl_{config["saved_target_model"]}',
        show_progress_bar=False
    )

    print('Model trained.')
            

