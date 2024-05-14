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
    output_directory = str(get_output_directory(Path(__file__).parent)) 
    directory_path = Path(output_directory+"/frequencies")
    for jsonl_file in directory_path.glob("frequency_*.jsonl"):
        print(jsonl_file.name)
        df = pd.read_json(jsonl_file, orient='records', lines=True)
        lang = re.split(r'[._]', jsonl_file.name)[1]
        letter_freq = dict(zip(df['letter'], df['frequency']))
        lang_freq.update({lang:letter_freq})

    
    print("Classification running...")
    prediction={}
    for id in tqdm(targets_validation.get("id")):
        text_processed = text_validation.query('id=="{}"'.format(id)).get("text").values[0]

        # calculate text relative frequencies
        letter_freq = {}
        text_processed = text_processed.lower()
        text_processed = re.sub(r'[0-9.,!;:-=@#$%^&*()_+€/?{}"\' ]','',text_processed)
        text_array = list(text_processed)
        for letter in text_array:
            if letter not in letter_freq:
                letter_freq.update({letter:1})
            else: 
                letter_freq[letter] = letter_freq[letter]+1
        for letter in letter_freq:
            letter_freq[letter]=letter_freq[letter]/len(text_array)


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
        
        if distances=={}:
            print("TEXT:"+text_processed)
            print(pd.Series(distances))
            continue
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
    """