# Data: `Unsupervised Domain adaptation for Sentence Classification`

This directory contains all of the data used during the course of this project. The main dataset we are focusing on is the SentiNews dataset, which is a collection of articles in Slovene, with differing domains, available here: [https://www.clarin.si/repository/xmlui/handle/11356/1495](https://www.clarin.si/repository/xmlui/handle/11356/1495). 

The SentiNet dataset is present in the [SentiNet](SentiNet/) directory and is split into two files: `slovenian_test.json` and `slovenian_train.json`. The files themselves are structured in such a way, that each line represents a valid JSON structure, containing the article body, title, keywords and language.