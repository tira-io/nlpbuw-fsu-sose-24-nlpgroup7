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
    
    #################### Training
    # getting the letters absolute frequencies
    i=1
    lang_freq = {}
    for id in targets_validation.get("id"):

        # get true lang of the text 
        lang = targets_validation.query('id=="{}"'.format(id)).get("lang").values[0]
        text_processed = text_validation.query('id=="{}"'.format(id)).get("text").values[0]
        if lang in lang_freq:
            letter_freq = lang_freq.get(lang)
        else:            
            letter_freq = {}
        text_processed = text_processed.lower()
        text_processed = re.sub(r'[0-9.,!;:-=@#$%^&*()_+€/?{}"\' ]','',text_processed)
        text_array = list(text_processed)
        for letter in text_array:
            if letter not in letter_freq:
                letter_freq.update({letter:1})
            else: 
                letter_freq[letter] = letter_freq[letter]+1
        
        # add to the common dictionary
        lang_freq.update({lang:letter_freq})

        # stop training
        i+=1
        if int(id)>10000 : 
            print("training set: "+str(i))
            last_id=int(id)
            print("last id: "+str(id))
            break
    
    # obtaining relative frequencies
    for lang in lang_freq.keys():
        summa = sum(lang_freq[lang].values())
        for letter in lang_freq[lang]:
            lang_freq[lang][letter]=lang_freq[lang][letter]/summa
    #print(pd.Series(lang_freq["ru"]).head(50))



    ################# classifying the data
    prediction={}
    for id in tqdm(targets_validation.get("id")):
        #if (int(id) <= last_id):
        #    continue
        text_processed = text_validation.query('id=="{}"'.format(id)).get("text").values[0]
        #print(text_processed)


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
        #print(pd.Series(letter_freq))


        # compare with trained data
        distances={}
        for lang in lang_freq.keys():
            dist=0
            coincidence=0
            for letter in lang_freq[lang]:
                if letter in letter_freq.keys():
                    coincidence+=1
                    dist += abs(letter_freq[letter]-lang_freq[lang][letter])
            #if (coincidence<5):
            #    continue
            distances.update({lang:dist})
        #print(pd.Series(distances))

        min_key = min(distances, key=distances.get)
        prediction.update({id:min_key})

    # converting the prediction to the required format
    prediction = list(prediction.items())
    prediction = pd.DataFrame(prediction)
    prediction.columns = ["id", "lang"]
    #print(prediction)

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions1.jsonl", orient="records", lines=True
    )
"""   
    prediction.name = "lang"
    prediction = prediction.to_frame()
    prediction["id"] = text_validation["id"]
    prediction = prediction[["id", "lang"]]

        break
     
   

    # classifying the data
    for lang_id in tqdm(lang_freq.keys()):
         


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
