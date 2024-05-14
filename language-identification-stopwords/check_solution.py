from pathlib import Path
import re

from tqdm import tqdm
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )


    #print("targets_validation:")
    #print(targets_validation.columns)
    
    #print(targets_validation.head())
    #print("text_validation:")
    #print(text_validation.columns)
    #print(text_validation.head())
    
    predictions = pd.read_json(path_or_buf=Path(__file__).parent / "predictions.jsonl", lines=True) 
    #print(predictions.query('id==14').get("lang"))
    #print(targets_validation.query('id=="14"').get("lang"))

    #print(targets_validation.query('id==14'))
    #print(predictions.get("id"))
    #print(predictions.query('id=={}'.format(14)).get("lang").to_string)
    #print(targets_validation.query('id=="{}"'.format(14)).get("lang").values[0])
    count = 0
    for i in tqdm(predictions.get("id")):
        if predictions.query('id=={}'.format(i)).get("lang").values[0] == targets_validation.query('id=="{}"'.format(i)).get("lang").values[0]:
            count=count+1
    print("Coincidence = {}".format(count))
    print("Accurtacy = {}".format(count/len(predictions)))
""" 


    stopwords = {
        lang_id: set(
            (Path(__file__).parent / "stopwords" / f"stopwords-{lang_id}.txt")
            .read_text()
            .splitlines()
        )
        - set(("(", ")", "*", "|", "+", "?"))  # remove regex special characters
        for lang_id in lang_ids
    }

    # classifying the data
    stopword_fractions = []
    for lang_id in tqdm(lang_ids):
        lang_stopwords = stopwords[lang_id]
        counts = pd.Series(0, index=text_validation.index, name=lang_id)
        for stopword in lang_stopwords:
            counts += (
                text_validation["text"]
                .str.contains(stopword, regex=False, case=False)
                .astype(int)
            )
        stopword_fractions.append(counts / len(lang_stopwords))
    stopword_fractions = pd.concat(stopword_fractions, axis=1)

    prediction = stopword_fractions.idxmax(axis=1)

    # converting the prediction to the required format
    prediction.name = "lang"
    prediction = prediction.to_frame()
    prediction["id"] = text_validation["id"]
    prediction = prediction[["id", "lang"]]

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
"""