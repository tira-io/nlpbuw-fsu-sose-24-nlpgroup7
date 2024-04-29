from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sklearn.metrics import accuracy_score


if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", f"authorship-verification-validation-20240408-training"
    )
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training")


    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    predictions = model.predict(df["text"])
    df["generated"] = predictions
    df = df[["id", "generated"]]

    # Calculate accuracy on validation/test data
    accuracy = accuracy_score(targets_validation['generated'], predictions)
    print("Accuracy:", accuracy)

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
