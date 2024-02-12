import nltk

nltk.download("stopwords")
nltk.download("wordnet")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import string

import json
import pandas

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


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

lemmatizer = WordNetLemmatizer()

def pre_processing(text):
  text_processed = text.translate(str.maketrans("","", string.punctuation))
  text_processed = text_processed.split()
  result = []
  for word in text_processed:
    word_processed = word.lower()
    if word_processed not in stopwords.words("english"):
      word_processed = lemmatizer.lemmatize(word_processed)
      result.append(word_processed)
  return result
  
count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)

data = count_vectorize_transformer.transform(data)

machine = MultinomialNB()
machine.fit(data,target)




new_reviews = pandas.read_csv("new_reviews.csv") 
new_reviews_tranformed = count_vectorize_transformer.transform(new_reviews.iloc[:,0])


prediction = machine.predict(new_reviews_tranformed)
prediction_prob = machine.predict_proba(new_reviews_tranformed)
print(prediction)
print(prediction_prob)

new_reviews['prediction'] = prediction
prediction_prob_dataframe = pandas.DataFrame(prediction_prob)


prediction_prob_dataframe = prediction_prob_dataframe.rename(columns={
  prediction_prob_dataframe.columns[0]: "prediction_prob_1",
  prediction_prob_dataframe.columns[1]: "prediction_prob_3",
  prediction_prob_dataframe.columns[2]: "prediction_prob_5"
  })



new_reviews = pandas.concat([new_reviews,prediction_prob_dataframe], axis=1)

print(new_reviews)


new_reviews = new_reviews.rename(columns={
  new_reviews.columns[0]: "text"
  })

new_reviews['prediction'] = new_reviews['prediction'].astype(int)
new_reviews['prediction_prob_1'] = round(new_reviews['prediction_prob_1'],4)
new_reviews['prediction_prob_3'] = round(new_reviews['prediction_prob_3'],4)
new_reviews['prediction_prob_5'] = round(new_reviews['prediction_prob_5'],4)


new_reviews.to_csv("new_reviews_with_prediction.csv", index=False)































