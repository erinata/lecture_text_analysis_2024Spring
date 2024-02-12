import nltk

nltk.download("stopwords")
nltk.download("wordnet")

import string

import json
import pandas

review_stars=[]
review_text=[]
with open("yelp_review_part.json", encoding="utf-8") as f:
  for line in f:
    json_line = json.loads(line)
    review_stars.append(json_line["stars"])
    review_text.append(json_line["text"])

# print(review_stars)
# print(review_text)

dataset = pandas.DataFrame(data={"text": review_text, "stars": review_stars})


dataset = dataset[0:3000]
dataset = dataset[(dataset['stars'] == 1) | (dataset['stars'] == 3) | (dataset['stars'] == 5)]

print(dataset)

data = dataset["text"]
target = dataset["stars"]















