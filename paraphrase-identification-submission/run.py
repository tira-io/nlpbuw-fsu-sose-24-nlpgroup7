from pathlib import Path
import pandas as pd
from joblib import load
from cosineSimilarity import calculate_cosine_similarity

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    # Load the data
    tira = Client()
    text_val = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")

    model = load(Path(__file__).parent / "model.joblib")

    # calculate cosine similarity for all pairs of sentences
    similarities = text_val.apply(lambda x: calculate_cosine_similarity(x['sentence1'], x['sentence2']), axis=1)

    # add similarity property to data
    text_val['similarity'] = similarities
    y_pred = model.predict(text_val.loc[:, ['similarity']])

    # Write predictions to file
    predicted = [{'id': id_, 'label': pred} for id_, pred in zip(text_val.index, y_pred)]

    # Convert predicted list to DataFrame
    predicted_df = pd.DataFrame(predicted)

    output_directory = get_output_directory(str(Path(__file__).parent))
    # Save DataFrame to JSON file
    predicted_df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)