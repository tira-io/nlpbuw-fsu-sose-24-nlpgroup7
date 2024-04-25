import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def write_answers(dataset_path: Path, predicted: list) -> None:
    """ Writes the given answers to a file compliant with the datasets format """
    open(Path(dataset_path) / 'truth.jsonl', 'w').writelines([json.dumps(line)+"\n" for line in predicted])

tira = Client()

# Loading train data
text_train = tira.pd.inputs("nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training")
targets_train = tira.pd.truths("nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training")

# Loading validation data (automatically replaced by test data when run on tira)
text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training")
targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training")

vectorizer = TfidfVectorizer()

# Fit the vectorizer on the training data and transform the text data into TF-IDF vectors
X_train = vectorizer.fit_transform(text_train['text'])

# Transform the validation/test data into TF-IDF vectors
X_validation = vectorizer.transform(text_validation['text'])

# Extract labels
y_train = targets_train['generated']
y_validation = targets_validation['generated']

# Train model
classifier = LogisticRegression() # Logistic Classifier with well interpretable results
classifier.fit(X_train, y_train)

# Predict on validation/test data
y_pred = classifier.predict(X_validation)

# Calculate accuracy on validation/test data
accuracy = accuracy_score(y_validation, y_pred)
print("Accuracy:", accuracy)

# Write predictions to file #TODO replace this by a better method
predicted = [{'id': id_, 'generated': str(pred)} for id_, pred in zip(text_validation['id'], y_pred)]
# write_answers("test/", predicted)
