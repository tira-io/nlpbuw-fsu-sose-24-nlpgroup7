from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from joblib import dump
from cosineSimilarity import calculate_cosine_similarity

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# Load the data
tira = Client()

text_train = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
).set_index("id")
labels_train = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
).set_index("id")

text_val= tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
).set_index("id")
labels_val = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
).set_index("id")

# Create a Pipeline for vectorizer and classifier
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

# calculate cosine similarity for all pairs of sentences
similarities = text_train.apply(lambda x: calculate_cosine_similarity(x['sentence1'], x['sentence2']), axis=1)
# add similarity property to data
text_train['similarity'] = similarities
# Fit the model
model.fit(text_train.loc[:, ['similarity']], labels_train.values.ravel())

# Save the model
output_directory = get_output_directory(str(Path(__file__).parent))
dump(model, Path(output_directory) / "model.joblib")



# calculate similarities for the validation set
similarities_val = text_val.apply(lambda x: calculate_cosine_similarity(x['sentence1'], x['sentence2']), axis=1)
text_val['similarity'] = similarities_val

# Predict on validation/test data
y_pred = model.predict(text_val.loc[:, ['similarity']])

# Calculate accuracy on validation/test data
accuracy = accuracy_score(labels_val['label'], y_pred)
print("Accuracy:", accuracy)

# Write predictions to file
predicted = [{'id': id_, 'label': pred} for id_, pred in zip(text_val.index, y_pred)]

# Convert predicted list to DataFrame
predicted_df = pd.DataFrame(predicted)

# Save DataFrame to JSON file
predicted_df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)