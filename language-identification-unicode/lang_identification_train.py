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
    lang_means = {}
    lang_deviations={}
    print("Training run...")
    for id in tqdm(targets_validation.get("id")):

        # get true lang of the text 
        lang = targets_validation.query('id=="{}"'.format(id)).get("lang").values[0]
        text_processed = text_validation.query('id=="{}"'.format(id)).get("text").values[0]
        text_processed = text_processed.lower()
        text_processed = re.sub(r'[0-9.,!;:-=@#$%^&*()_+€/?{}"\' ]','',text_processed)
        if len(text_processed) == 0:
            continue

        if lang not in lang_means.keys():
            lang_means.update({lang:[]})
            lang_deviations.update({lang:[]})
        
        #calculate means
        unicode_codes = [ord(char) for char in text_processed]
        mean_value = sum(unicode_codes) / len(unicode_codes)
        lang_means[lang].append(mean_value)

        #calculate deviations
        squared_diffs = [(code - mean_value) ** 2 for code in unicode_codes]
        unicode_codes = [ord(char) for char in text_processed]
        mean_squared_diff = sum(squared_diffs) / len(squared_diffs)
        lang_deviations[lang].append(mean_squared_diff)

        # stop training
        # i+=1
        # if int(id)>10000 : 
        #     print("training set: "+str(i))
        #     last_id=int(id)
        #     print("last id: "+str(id))
        #     break
        
        
    for lang in lang_means.keys():
        lang_means[lang] = sum(lang_means[lang]) / len(lang_means[lang])
        lang_deviations[lang] = sum(lang_deviations[lang]) / len(lang_deviations[lang])
    print(pd.concat([pd.Series(lang_deviations),pd.Series(lang_means)],axis = 1))
    

    
    # saving the means and deviations
    lang_means=pd.DataFrame(list(lang_means.items()), columns=['lang', 'mean'])
    lang_deviations=pd.DataFrame(list(lang_deviations.items()), columns=['lang', 'deviation'])
    output_directory = get_output_directory(str(Path(__file__).parent))
    lang_means.to_json(
        Path(output_directory) / "means.jsonl".format(lang), orient="records", lines=True, force_ascii=False
    )
    lang_deviations.to_json(
        Path(output_directory) / "deviations.jsonl".format(lang), orient="records", lines=True, force_ascii=False
    )
