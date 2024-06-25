from pathlib import Path
import spacy
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from seqeval.metrics import classification_report
import pandas as pd

def predict(texts):
    predictions = []
    for id_, text in zip(texts.index, texts):
        doc = nlp(text)
        tags = ["O"] * len(doc)
        for ent in doc.ents:
            tags[ent.start] = f"B-{ent.label_}"
            for i in range(ent.start + 1, ent.end):
                tags[i] = f"I-{ent.label_}"
        predictions.append({"id": id_, "tags": tags})
    return predictions


if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    ).set_index("id")
    
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )

    nlp = spacy.load(str(Path(__file__).parent)+"/model")
    predictions = predict(text_validation['sentence'])
    # print(predictions)
   
#     new_texts = [
#     "Belarus’s Foreign Ministry has dismissed the latest U.S.-imposed sanctions aimed at freezing the U.S. assets of Belarusian President Alexander Lukashenko.",
#     "The new Prime Minister of the United Kingdom, Boris Johnson, gave a speech."
# ]

# # Process each new text and print the recognized entities
# for text in new_texts:
#     doc = nlp(text)
#     print(f"Text: {text}")
#     print("Entities:")
#     for ent in doc.ents:
#         print(f"  - {ent.text} ({ent.label_})")
#     print()

    predictions_df = pd.DataFrame(predictions)
    print(predictions_df)

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions_df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )