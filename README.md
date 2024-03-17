# Natural language processing course 2023/24: `Unsupervised Domain adaptation for Sentence Classification`

This project aims to fine-tune the SBERT model on specific domains using the TSDAE and GPL fine-tuning models. Specifically, we attempt to fine-tune the model on Slovene articles pertaining to keywords "economics", "economy" and "bank", thereby adapting the model to the domain of economics. 

## Project structure

The project is structured in the following way:

- The [data](data/) directory contains the datasets we used to train and test our models. The main dataset we used is the [SentiNews](https://www.clarin.si/repository/xmlui/handle/11356/1397#) dataset.
- The [report](report/) directory contains various files pertaining to the report of this project, including figures, `.tex` files and the compiled `.pdf` version of the report.
- The [src](src/) directory contains the complete implementation of the project.

## Installation and running

To run this project, we recommend setting up either a Python virtual environment, or a Conda virtual environment (Python version 3.10 is recommended) and installing the necessary packages, which are listed in the [requirements.txt](requirenements.txt) file.

