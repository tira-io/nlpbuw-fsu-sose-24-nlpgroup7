from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
import nltk
nltk.download('punkt') 
import gensim.downloader
import spacy
from nltk.tokenize import word_tokenize

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

nlp = spacy.load('en_core_web_md')

# Load the data
tira = Client()
df = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
).set_index("id")
labels = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
).set_index("id")

model = api.load('word2vec-google-news-300')

def tokenize(text):
    return [word.lower() for word in word_tokenize(text)]

def calculate_wmd(sentence1, sentence2):
    wmd_distance = model.wmdistance(tokenize(sentence1), tokenize(sentence2))
    return wmd_distance

distances = df.apply(lambda x: calculate_wmd(x['sentence1'], x['sentence2']), axis=1)

df['distance'] = distances
df['label'] = (df['distance'] >= 0.9).astype(int)

df = df.drop(columns=['distance', 'sentence1', 'sentence2']).reset_index()

print(df)
accuracy = accuracy_score(labels['label'], df['label'])
print("Accuracy:", accuracy)