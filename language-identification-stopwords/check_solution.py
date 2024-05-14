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

    
    predictions = pd.read_json(path_or_buf=Path(__file__).parent / "predictions1.jsonl", lines=True) 
    count = 0
    for i in tqdm(predictions.get("id")):
        if predictions.query('id=={}'.format(i)).get("lang").values[0] == targets_validation.query('id=="{}"'.format(i)).get("lang").values[0]:
            count=count+1
    print("Coincidence = {}".format(count))
    print("Accurtacy = {}".format(count/len(predictions)))
