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

    ################# classifying the data

    # Read the frequencies into a DataFrame
    lang_freq = {}
    output_directory = str(Path(__file__).parent)
    means_file_path = Path(output_directory) / "means.jsonl"
    deviations_file_path = Path(output_directory) / "deviations.jsonl"

    lang_means = pd.read_json(means_file_path, orient='records', lines=True)
    lang_deviations = pd.read_json(deviations_file_path, orient='records', lines=True)
    # print(pd.concat([lang_means,lang_deviations],axis = 1))
    
    lang_means=lang_means.set_index('lang').T.to_dict()
    lang_deviations=lang_deviations.set_index('lang').T.to_dict()

    print("Classification running...")
    prediction={}
    for id in tqdm(targets_validation.get("id")):
        text_processed = text_validation.query('id=="{}"'.format(id)).get("text").values[0]

        # process text
        letter_freq = {}
        text_processed = text_processed.lower()
        text_processed = re.sub(r'[0-9.,!;:-=@#$%^&*()_+€/?{}"\' ]','',text_processed)
        if len(text_processed) == 0:
            continue
        
        #calculate mean
        unicode_codes = [ord(char) for char in text_processed]
        mean_value = sum(unicode_codes) / len(unicode_codes)

        #calculate deviation
        squared_diffs = [(code - mean_value) ** 2 for code in unicode_codes]
        unicode_codes = [ord(char) for char in text_processed]
        mean_squared_diff = sum(squared_diffs) / len(squared_diffs)

        # compare with trained means and deviations
        distances = {}
        # print(text_processed)
        for lang in lang_means.keys():
            tr_mean = lang_means[lang]["mean"]
            tr_dev = lang_deviations[lang]["deviation"]
            distance_mean = abs(float(tr_mean) - mean_value)
            distance_dev = abs(float(tr_dev)-mean_squared_diff)
            distance = distance_mean + distance_dev/8000
            distances.update({lang:distance})
            # print(lang)
            # print("lang: "+str(tr_mean) +'  '+ str(tr_dev))
            # print("real: "+str(mean_value) +'  '+ str(mean_squared_diff))
            # print(distance)
        min_key = min(distances, key=distances.get)
        prediction.update({id:min_key})

    # converting the prediction to the required format
    prediction = list(prediction.items())
    prediction = pd.DataFrame(prediction)
    prediction.columns = ["id", "lang"]

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

"""
        # compare with trained data
        distances={}
        for lang in lang_freq.keys():
            dist=0
            coincidence=0
            for letter in lang_freq[lang]:
                if letter in letter_freq.keys():
                    coincidence+=1
                    dist += abs(letter_freq[letter]-lang_freq[lang][letter])
            distances.update({lang:dist})
        
        min_key = min(distances, key=distances.get)
        prediction.update({id:min_key})

    # converting the prediction to the required format
    prediction = list(prediction.items())
    prediction = pd.DataFrame(prediction)
    prediction.columns = ["id", "lang"]

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
"""