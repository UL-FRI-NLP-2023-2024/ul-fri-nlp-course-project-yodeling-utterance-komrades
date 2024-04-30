# Natural language processing course 2023/24: `Unsupervised Domain adaptation for Sentence Classification`

This project aims to fine-tune the SBERT model on specific domains using the TSDAE and GPL fine-tuning models. Specifically, we attempt to fine-tune the model on Slovene articles pertaining to keywords "economics", "economy" and "bank", thereby adapting the model to the domain of economics. 

## Project structure

The project is structured in the following way:

- The [data](data/) directory contains the datasets we used to train and test our models. The main dataset we used is the [SentiNews](https://www.clarin.si/repository/xmlui/handle/11356/1397#) dataset.
- The [models](models/) directory contains the models produced by the training procedure.
- The [report](report/) directory contains various files pertaining to the report of this project, including figures, `.tex` files and the compiled `.pdf` version of the report. The [report.pdf](report/report.pdf) file contains the rolling version of the project report.
- The [src](src/) directory contains the complete implementation of the project.

## Installation and running

### Local setup

To set up this project on a local machine, a CUDA capable GPU is required. If you wish to set up the project locally, follow the steps below:

1. Clone this repository into your desired location
2. Install the required packages (available in the `requirements.txt` file in the [src](src/) directory)
3. Run `python baseline_run.py`

### HPC setup

To set up this project on the Arnes HPC, follow the steps below:

1. Clone this repository into your desired location
2. In the [src](src/) directory create a directory called `containers`.
3. Move into the `containers` directory and run the following command: `singularity build ./container-torch.sif docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel`
4. Install the required packages (available in the `requirements.txt` file in the [src](src/) directory)
    - The packages must be installed using the following command `singularity exec ./containers/container-torch.sif pip install <package-name>`
    - Not all packages from the `requirements.txt` file must be installed. If installing by hand, simply install sentence-transformers, scikit-learn and numpy.  
5. Run `sbatch sbatch_run.sh`

Additional information about possible modifications is found in the README.md file in the [src](src/) directory.