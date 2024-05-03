# Sources: `Unsupervised Domain adaptation for Sentence Classification`

This directory contains the entire source code of this project. 

## Baseline model

The first part of this project was creating a simple classification head for a sentence transformer, to be used to classify the embeddings of articles produced by the sentence transformer into keywords. 

The neural network classification head is located in the `networks/baseline_classifier.py`. Additional utility scripts and our custom DataLoader implementation are present in the `utils` directory.

The driver script of this part of the project is the `baseline_run.py` file. It contains the entire training and evaluation code. To modify the training, testing or the classification head parameters or the sentence transformer, simply modify the appropriate valzes in the `config` dictionary located at the top of the file.
