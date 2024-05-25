from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
import pandas as pd
import json

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

nlp = spacy.load('en_core_web_md')

# Load the data
tira = Client()
text = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
).set_index("id")
labels = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
).set_index("id")

def get_word_vectors(sentence):
    tokens = nltk.word_tokenize(sentence)
    return np.array([nlp.vocab[token].vector for token in tokens])

def calculate_cosine_similarity(sentence1, sentence2):
    word_vectors1 = get_word_vectors(sentence1)
    word_vectors2 = get_word_vectors(sentence2)
    embedding1 = np.mean(word_vectors1, axis=0)
    embedding2 = np.mean(word_vectors2, axis=0)
    return cosine_similarity([embedding1], [embedding2])[0][0]

similarities = text.apply(lambda x: calculate_cosine_similarity(x['sentence1'], x['sentence2']), axis=1)

text['similarity'] = similarities
text['label'] = (text['similarity'] >= 0.9121241569519043).astype(int)

text = text.drop(columns=['similarity', 'sentence1', 'sentence2']).reset_index()

print(text)

accuracy = accuracy_score(labels['label'], text['label'])
print("Accuracy:", accuracy)

# df = text.join(labels)

# mccs = {}
# for threshold in sorted(text["similarity"].unique()):
#     tp = df[(df["similarity"] >= threshold) & (df["label"] == 1)].shape[0]
#     fp = df[(df["similarity"] >= threshold) & (df["label"] == 0)].shape[0]
#     tn = df[(df["similarity"] < threshold) & (df["label"] == 0)].shape[0]
#     fn = df[(df["similarity"] < threshold) & (df["label"] == 1)].shape[0]
#     try:
#         mcc = (tp * tn - fp * fn) / (
#             (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
#         ) ** 0.5
#     except ZeroDivisionError:
#         mcc = 0
#     mccs[threshold] = mcc
# best_threshold = max(mccs, key=mccs.get)
# print(f"Best threshold: {best_threshold}")

def jsonl_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse JSON object from each line
            json_object = json.loads(line)
            data.append(json_object)
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)
    return df

# Example usage:
file_path = '\predictions.jsonl'
df = jsonl_to_dataframe(file_path)
print(df.head())