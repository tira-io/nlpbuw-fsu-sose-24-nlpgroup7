from tira.rest_api_client import Client   
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import spacy
import matplotlib.pyplot as plt

# load spacy model for sentence embedding
nlp = spacy.load('en_core_web_md')

 # Load the data
tira = Client()
text = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
).set_index("id")
labels = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
).set_index("id")

def get_word_vectors(sentence):
    doc = nlp(sentence)
    return np.array([token.vector for token in doc])

# Function to calculate cosine similarity between two sentences
def calculate_cosine_similarity(sentence1, sentence2):
    word_vectors1 = get_word_vectors(sentence1)
    word_vectors2 = get_word_vectors(sentence2)
    embedding1 = np.mean(word_vectors1, axis=0)
    embedding2 = np.mean(word_vectors2, axis=0)
    if embedding1 is not None and embedding2 is not None:
        return cosine_similarity([embedding1], [embedding2])[0][0]
    else:
        return None

similarities = text.apply(lambda row: calculate_cosine_similarity(row['sentence1'], row['sentence2']), axis=1)
# Attach similarity scores to the DataFrame
text['similarity'] = similarities
text['label'] = (text['similarity'] >= 0.9121241569519043).astype(int)

text = text.drop(columns=['similarity', 'sentence1', 'sentence2']).reset_index()

accuracy = accuracy_score(labels['label'], text['label'])
print("Accuracy:", accuracy)

# # Print the DataFrame with similarity scores attached
# print(text)

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