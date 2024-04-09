import torch
import numpy as np
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from classifier import Classifier
from data_loader import DataLoader

if __name__ == '__main__':
    model = SentenceTransformer('distiluse-base-multilingual-cased')


    data_loader_train = DataLoader('../data/SentiNews/slovenian_train.json')
    data_train, labels_train = data_loader_train.load_data()

    data_loader_test = DataLoader('../data/SentiNews/slovenian_test.json')
    data_test, labels_test = data_loader_test.load_data()

    #embeddings = model.encode(data)

    #X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    embeddings_train = model.encode(data_train)
    embeddings_test = model.encode(data_test)

    X_train = torch.tensor(embeddings_train)
    y_train = torch.tensor(labels_train)

    X_test = torch.tensor(embeddings_test)
    y_test = torch.tensor(labels_test)

    input_dim = embeddings_train.shape[1]
    hidden_dim = 256

    classifier = Classifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    epochs = 2

    # Linear learning rate scheduler
    lmbda = lambda epoch: 1 - epoch / epochs
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = classifier(X_train)

        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}')
