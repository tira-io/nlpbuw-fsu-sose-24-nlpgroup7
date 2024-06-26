import random
from spacy.training import Example
from pathlib import Path
import spacy
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import tqdm
from spacy.util import minibatch, compounding
import re

nlp = spacy.blank("en")
    
def preprocess_data(texts, labels):
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    training_data = []
    for _, text_row in texts.iterrows():
        text_id = text_row['id']
        text_sentence = text_row['sentence']
        label_row = labels[labels['id'] == text_id]
        if not label_row.empty:
            tags = label_row['tags'].values[0]
            training_data.append({
                "id": text_id,
                "sentence": text_sentence,
                "tags": tags
            })           
                
    # add labels to pipeline
    for item in training_data:
        tags = item['tags']
        for tag in tags:
            if tag != "O":
                ner.add_label(tag.split("-")[1])
    return training_data

def char_index(sentence, word_index):
    sentence = re.split('(\s)',sentence) # parentheses keep split characters
    return len(''.join(sentence[:word_index*2]))

def train(training_data, n_iter): 
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for _ in tqdm.tqdm(range(n_iter)):
            losses = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                for data in batch:
                    text = data['sentence']
                    tags = data['tags']
                    doc = nlp.make_doc(text)
                    entities = []
                    entity_start = None
                    entity_type = None
                    for i, tag in enumerate(tags):
                        if tag.startswith("B-"):
                            if entity_start is not None:
                                entities.append((entity_start, entity_end, entity_type))
                            entity_start = char_index(text, i)
                            entity_end = entity_start + len(text.split()[i])
                            entity_type = tag.split("-")[1]
                        elif tag.startswith("I-") and entity_type:
                            continue
                        else:
                            if entity_start is not None:
                                entity_end = char_index(text, i-1) + len(text.split()[i-1])
                                entities.append((entity_start, entity_end, entity_type))
                                entity_start = None
                                entity_type = None

                    if entity_start is not None:
                        entities.append((entity_start, entity_start + len(text.split()[len(tags)-1]), entity_type))
                    
                example = Example.from_dict(doc, {"entities": entities})
                nlp.update(
                    [example],
                    drop=0.2,
                    sgd=optimizer,
                    losses=losses)
            print(losses)
                
def save_model(output_dir=None):
# save trained model
    if output_dir is not None:
        output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
    
    
    
def print_training_data_sample(training_data, num_samples=5):
    for i, data in enumerate(training_data[:num_samples]):
        print(f"Sample {i+1}:")
        print(f"Text: {data['sentence']}")
        print(f"Tags: {data['tags']}")
        print()
    
    
if __name__ == "__main__":

    tira = Client()

    # loading training data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-training-20240612-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-training-20240612-training"
    )
    
    output_directory = get_output_directory(str(Path(__file__).parent) + "/model_new")

    training_data = preprocess_data(text_train, targets_train)
    # print_training_data_sample(training_data)
    
    train(training_data, 30)
    
    save_model(output_dir=output_directory)