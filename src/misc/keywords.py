# Description: This script reads the Slovenian training/test data and counts the frequency of each keyword.
# It is used to find the most common keywords in the dataset, which helps us choose the domain on which
# we train the model.

import json

f = open('../../data/SentiNews/slovenian_train.json', 'r', encoding='utf-8')

keywords = dict()

for line in f.readlines():
    data = json.loads(line)
    article_keywords = data['keywords'].split(';')
    for keyword in article_keywords:
        keyword = keyword.strip().lower()
        if keyword in keywords:
            keywords[keyword] += 1
        else:
            keywords[keyword] = 1

f.close()

sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
print(sorted_keywords)