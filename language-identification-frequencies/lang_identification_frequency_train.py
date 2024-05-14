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
    print("Training run...")
    for id in tqdm(targets_validation.get("id")):

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

        #stop training
        # i+=1
        # if int(id)>10000 : 
        #     print("training set: "+str(i))
        #     last_id=int(id)
        #     print("last id: "+str(id))
        #     break

    # filter common letters - DOESNT WORK
    # for lang in lang_freq.keys():
    #     dict1 = lang_freq[lang]
    #     for lang1 in lang_freq.keys():
    #         if lang == lang1:
    #             continue
    #         dict2 = lang_freq[lang1]
    #         for key in list(dict1.keys()):
    #             if key in dict2:
    #                 dict1.pop(key)

    
    # obtaining relative frequencies
    for lang in lang_freq.keys():
        summa = sum(lang_freq[lang].values())
        for letter in lang_freq[lang]:
            lang_freq[lang][letter]=lang_freq[lang][letter]/summa
    
    # saving the frequencies
    for lang in lang_freq.keys():
        lang_freq_pd=pd.DataFrame(list(lang_freq[lang].items()), columns=['letter', 'frequency'])
        output_directory = get_output_directory(str(Path(__file__).parent))
        lang_freq_pd.to_json(
            Path(output_directory) / "frequencies"/"frequency_{}.jsonl".format(lang), orient="records", lines=True, force_ascii=False
        )
