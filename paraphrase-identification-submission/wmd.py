from pathlib import Path
from sklearn.metrics import accuracy_score
import gensim.downloader as api
from nltk.tokenize import word_tokenize

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# load the model for Word Mover Distance calculation
model = api.load('word2vec-google-news-300')

# Load the data
tira = Client()
text = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
).set_index("id")
labels = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
).set_index("id")

def preprocess(sentence):
    return [word.lower() for word in word_tokenize(sentence)]

def calculate_wmd(sentence1, sentence2):
    word_vectors1 = preprocess(sentence1)
    word_vectors2 = preprocess(sentence2)
    distance = model.wmdistance(word_vectors1, word_vectors2)
    return distance

# calculate cosine similarity for all pairs of sentences
distances = text.apply(lambda x: calculate_wmd(x['sentence1'], x['sentence2']), axis=1)

# add similarity property to data
text['distance'] = distances

df = text.join(labels)
# calculate the Mattews Correlation
mccs = {}
for threshold in sorted(text["distance"].unique()):
    tp = df[(df["distance"] <= threshold) & (df["label"] == 1)].shape[0]
    fp = df[(df["distance"] <= threshold) & (df["label"] == 0)].shape[0]
    tn = df[(df["distance"] > threshold) & (df["label"] == 0)].shape[0]
    fn = df[(df["distance"] > threshold) & (df["label"] == 1)].shape[0]
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
text['label'] = (text['distance'] <= best_threshold).astype(int)
text = text.drop(columns=['distance', 'sentence1', 'sentence2']).reset_index()

accuracy = accuracy_score(labels['label'], text['label'])
print("Accuracy:", accuracy)

# Save the predictions
output_directory = get_output_directory(str(Path(__file__).parent))
text.to_json(
    Path(output_directory) / "predictions.jsonl", orient="records", lines=True
)