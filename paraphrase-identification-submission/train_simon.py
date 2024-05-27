from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

nlp = spacy.load('en_core_web_md')

# Load the data
tira = Client()
text = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
).set_index("id")
labels = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
).set_index("id")

# preprocess sentences by removing punctuation and making everything lowercase
def preprocess(sentence):
    sentence = sentence.lower()
    # get rid of punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

# compute word vectors for words in sentences
def get_word_vectors(sentence):
    doc = nlp(preprocess(sentence))
    return np.array([token.vector for token in doc])

def calculate_cosine_similarity(sentence1, sentence2):
    word_vectors1 = get_word_vectors(sentence1)
    word_vectors2 = get_word_vectors(sentence2)
    # compute average of all word vectors for each sentence
    embedding1 = np.mean(word_vectors1, axis=0)
    embedding2 = np.mean(word_vectors2, axis=0)
    # compute cosine similarity between averaged embeddings
    return cosine_similarity([embedding1], [embedding2])[0][0]

# calculate cosine similarity for all pairs of sentences
similarities = text.apply(lambda x: calculate_cosine_similarity(x['sentence1'], x['sentence2']), axis=1)

# add similarity property to data
text['similarity'] = similarities

df = text.join(labels)
# calculate the Mattews Correlation
mccs = {}
for threshold in sorted(text["similarity"].unique()):
    tp = df[(df["similarity"] >= threshold) & (df["label"] == 1)].shape[0]
    fp = df[(df["similarity"] >= threshold) & (df["label"] == 0)].shape[0]
    tn = df[(df["similarity"] < threshold) & (df["label"] == 0)].shape[0]
    fn = df[(df["similarity"] < threshold) & (df["label"] == 1)].shape[0]
    try:
        mcc = (tp * tn - fp * fn) / (
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        ) ** 0.5
    except ZeroDivisionError:
        mcc = 0
    mccs[threshold] = mcc
best_threshold = max(mccs, key=mccs.get)
print(f"Best threshold: {best_threshold}")

# use the threshold to classify examples
text['label'] = (text['similarity'] >= best_threshold).astype(int)
text = text.drop(columns=['similarity', 'sentence1', 'sentence2']).reset_index()

accuracy = accuracy_score(labels['label'], text['label'])
print("Accuracy:", accuracy)

# Save the predictions
output_directory = get_output_directory(str(Path(__file__).parent))
text.to_json(
    Path(output_directory) / "predictions.jsonl", orient="records", lines=True
)