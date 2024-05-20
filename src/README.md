# Sources: `Unsupervised Domain adaptation for Sentence Classification`

This directory contains the entire source code of this project. 

## Baseline model

The aim of this project is to provide implementations to the TSDAE and GPl fine-tuning methods for fine-tuning sentence transformers for the task of classifying articles into their sentiments and keywords. To achieve this goal, multiple python scripts have been developed.

### Driver scripts

There are multiple driver scripts, used to run certain parts of the project:

- `baseline_run.py`: This script is deprecated. It's purpose was to train and evaluate multiple sentence transformers in order to discover which would prove to be the best baseline model to finetune. This script only performs multi-label classification, not multi-task. 
- `multi_task_run.py`: This is the main training/testing script. It's purpose is to load a sentence transformer and train/test a classification head. All training, testing and network parameters can be modified by simply modifying the config dictionary.
- `tsdae.py`: This script executes the TSDAE fine-tuning method on the `sentence-transformers/LaBSE` model, storing it and it's training configuration in the `../models` directory.
- `gpl.py`: This script executes the GPL fine-tuning method on the `sentence-transformers/LaBSE` model, storing it and it's training configuration in the `../models` directory.
