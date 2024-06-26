from pathlib import Path
import spacy
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from seqeval.metrics import classification_report
import pandas as pd
import re
from spacy.tokens import Doc

def predict(texts, vocab):
    predictions = []
    for id_, text in zip(texts.index, texts):
        words = re.findall(r'\S+|\u0085|\u0094', text) # split words and include special operators
        doc_ = Doc(vocab, words) # build doc for the purpose of correct number of tokens for initialization of "tags"
        tags = ["O"] * len(doc_)
        doc = nlp(doc_) # classify entities based on the tokenized input
        for ent in doc.ents:
            # write output according to the IOB standard
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
    
    predictions = predict(text_validation['sentence'], nlp.vocab)
    
    predictions_df = pd.DataFrame(predictions)
    
    
    # # Check if the sizes of prediction labels and true labels match
    # for i in range(len(targets_validation['tags'])):
    #     if len(targets_validation['tags'][i]) != len(predictions_df['tags'][i]):
    #         print(f"Inconsistent lengths at index {i}: "
    #             f"targets length: {len(targets_validation['tags'][i])}, "
    #             f"predictions length: {len(predictions_df['tags'][i])}")
        
    # print(classification_report(targets_validation['tags'], predictions_df['tags'], zero_division=0))

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions_df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )