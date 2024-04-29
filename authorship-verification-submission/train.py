import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump

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

# Save the model
output_directory = get_output_directory(str(Path(__file__).parent))
dump(classifier, Path(output_directory) / "model.joblib")

# Predict on validation/test data
y_pred = classifier.predict(X_validation)

# Calculate accuracy on validation/test data
accuracy = accuracy_score(y_validation, y_pred)
print("Accuracy:", accuracy)

# Write predictions to file
predicted = [{'id': id_, 'generated': pred} for id_, pred in zip(text_validation['id'], y_pred)]

# Convert predicted list to DataFrame
predicted_df = pd.DataFrame(predicted)

# Save DataFrame to JSON file
predicted_df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
