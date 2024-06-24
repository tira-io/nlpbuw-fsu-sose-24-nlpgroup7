import json
import pandas as pd
import spacy
from spacy.training import Example
from seqeval.metrics import classification_report
from tira.rest_api_client import Client

tira = Client()


# Function to load JSONL data
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    return pd.DataFrame(data)

# Load train, validation, and test data
# train_text = load_jsonl('train_text.jsonl')
# train_labels = load_jsonl('train_labels.jsonl')
valid_text = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
valid_labels = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
# test_text = load_jsonl('test_text.jsonl')

# Merge text and labels
# train_data = pd.merge(train_text, train_labels, on='id')
valid_data = pd.merge(valid_text, valid_labels, on='id')

# Preprocess the data
def preprocess_data(data):
    formatted_data = []
    for _, row in data.iterrows():
        sentence = row['sentence']
        tags = row['tags']
        entities = []
        words = sentence.split(' ')
        for i, word in enumerate(words):
            if tags[i] != 'O':
                tag, entity = tags[i].split('-')
                start_char = sentence.find(word)
                end_char = start_char + len(word)
                entities.append((start_char, end_char, entity))
        formatted_data.append((sentence, {'entities': entities}))
    return formatted_data

# train_formatted = preprocess_data(train_data)
valid_formatted = preprocess_data(valid_data)

# Train the NER Model
nlp = spacy.blank('en')
ner = nlp.add_pipe('ner', last=True)

for _, annotations in valid_formatted:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

optimizer = nlp.begin_training()
for itn in range(20):
    losses = {}
    for text, annotations in valid_formatted:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, losses=losses)
    print(f"Iteration {itn}, Losses: {losses}")

# Save the model
nlp.to_disk('custom_ner_model')

# Load the trained model
nlp = spacy.load('custom_ner_model')

# Evaluate the model
def evaluate_model(nlp, data):
    true_entities = []
    pred_entities = []
    for text, annotations in data:
        doc = nlp(text)
        true_entities.extend(annotations.get('entities'))
        pred_entities.extend([(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])
    return true_entities, pred_entities

true_entities, pred_entities = evaluate_model(nlp, valid_formatted)

true_labels = [[ent[2] for ent in true_entities]]
pred_labels = [[ent[2] for ent in pred_entities]]

print(classification_report(true_labels, pred_labels))

# Make predictions on the test data
def make_predictions(nlp, data):
    predictions = []
    for _, row in data.iterrows():
        doc = nlp(row['sentence'])
        tags = ['O'] * len(row['sentence'].split(' '))
        for ent in doc.ents:
            start = row['sentence'][:ent.start_char].count(' ')
            end = row['sentence'][:ent.end_char].count(' ')
            tags[start] = f'B-{ent.label_}'
            for i in range(start + 1, end + 1):
                tags[i] = f'I-{ent.label_}'
        predictions.append({'id': row['id'], 'tags': tags})
    return predictions

test_predictions = make_predictions(nlp, test_text)

# Save predictions to predictions.jsonl
with open('predictions.jsonl', 'w', encoding='utf-8') as f:
    for pred in test_predictions:
        json.dump(pred, f)
        f.write('\n')
