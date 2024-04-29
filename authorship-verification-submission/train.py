import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from joblib import dump

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

tira = Client()

# Loading train data
text_train = tira.pd.inputs("nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training")
targets_train = tira.pd.truths("nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training")

# Loading validation data (automatically replaced by test data when run on tira)
text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training")
targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training")

# Create a Pipeline for vectorizer and classifier
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

# Fit the model
model.fit(text_train['text'], targets_train['generated'])

# Save the model
output_directory = get_output_directory(str(Path(__file__).parent))
dump(model, Path(output_directory) / "model.joblib")

# Predict on validation/test data
y_pred = model.predict(text_validation['text'])

# Calculate accuracy on validation/test data
accuracy = accuracy_score(targets_validation['generated'], y_pred)
print("Accuracy:", accuracy)

# Write predictions to file
predicted = [{'id': id_, 'generated': pred} for id_, pred in zip(text_validation['id'], y_pred)]

# Convert predicted list to DataFrame
predicted_df = pd.DataFrame(predicted)

# Save DataFrame to JSON file
predicted_df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
