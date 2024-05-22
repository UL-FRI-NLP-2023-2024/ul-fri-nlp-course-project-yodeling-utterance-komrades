# Natural language processing course 2023/24: `Unsupervised Domain adaptation for Sentence Classification`

This project aims to fine-tune the SBERT model on specific domains using the TSDAE and GPL fine-tuning models. Specifically, we attempt to fine-tune the model on Slovene articles, aiming to increase the classification performance on the multi-task classification problem of classifying keywords and sentiment. 

## Project structure

The project is structured in the following way:

- The [data](data/) directory contains the datasets we used to train and test our models. The original dataset we used is the [SentiNews](https://www.clarin.si/repository/xmlui/handle/11356/1397#) dataset, before switching to the SentiNews dataset containing both keywords and sentiment.
- The [models](models/) directory contains the models produced by the training procedure.
- The [report](report/) directory contains various files pertaining to the report of this project, including figures, `.tex` files and the compiled `.pdf` version of the report. The [report.pdf](report/report.pdf) file contains the rolling version of the project report.
- The [src](src/) directory contains the complete implementation of the project.

## Installation and running

### Local setup

To set up this project on a local machine, a CUDA capable GPU is required. If you wish to set up the project locally, follow the steps below:

1. Clone this repository into your desired location
2. Install the required packages (available in the `requirements.txt` file in the [src](src/) directory)
3. Run one of the following files:
    - `python baseline_run.py` to run the baseline training/testing code. This will only train/test the baseline classifier.
    - `python multi_task_run.py` to run the multi task classifier training/testing code.
    - `python tsdae.py` to run the TSDAE finetuning method.
    - `python gpl.py` to run the GPL finetuning method.

### HPC setup

To set up this project on the Arnes HPC, follow the steps below:

1. Clone this repository into your desired location
2. In the [src](src/) directory create a directory called `containers`.
3. Move into the `containers` directory and run the following command: `singularity build ./container-torch.sif docker://pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`
4. Install the required packages (available in the `requirements.txt` file in the [src](src/) directory)
    - The packages must be installed using the following command `singularity exec ./containers/container-torch.sif pip install <package-name>`
    - Not all packages from the `requirements.txt` file must be installed. If installing by hand, simply install sentence-transformers, scikit-learn, numpy, nltk and tqdm.  
5. Run one of the following files:
    - `sbatch sbatch_run.sh` to run the baseline training/testing code. This will only train/test the baseline classifier.
    - `sbatch sbatch_run_multi_task.sh.py` to run the multi task classifier training/testing code.
    - `sbatch sbatch_run_tsdae.sh` to run the TSDAE finetuning method.
    - `sbatch sbatch_run_gpl.sh` to run the GPL finetuning method.
    - NOTE: If you want to run the TSDAE or GPL finetuning methods, you will have to manually download the nltk 'punkt' dataset and place it in your home folder:
        - First, create a directory called `nltk_data` in your home folder.
        - Next, download and extract the nltk 'punkt' dataset, available here: [https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip).
        - In the `nltk_data` directory, add a subdirectory called `tokenizers`.
        - Place the punkt directory you downloaded into the `nltk_data/tokenizers/` directory.

Additional information about possible modifications is found in the README.md file in the [src](src/) directory.
