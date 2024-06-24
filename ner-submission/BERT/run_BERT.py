import json
import numpy as np
import pandas as pd
from pathlib import Path
from tira.rest_api_client import Client
from transformers import BertTokenizer, BertForTokenClassification, Trainer

def load_data(text_data):
    texts = text_data.to_dict(orient='records')
    data = [{'id': text['id'], 'sentence': text['sentence'].split()} for text in texts]
    return data

def tokenize_sentences(examples):
    return tokenizer(examples['sentence'], truncation=True, is_split_into_words=True)

if __name__ == "__main__":
    tira = Client()

    # Load validation/test data
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")

    val_data = load_data(text_validation)

    model_name = "./saved_model"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)

    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data)).map(tokenize_sentences, batched=True)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
    )

    # Make predictions
    predictions, labels, _ = trainer.predict(val_dataset)
    predictions = np.argmax(predictions, axis=2)

    # Save predictions
    output = []
    for i, pred in enumerate(predictions):
        tags = [model.config.id2label[p] for p in pred if p != -100]
        output.append({"id": val_data[i]['id'], "tags": tags})

    output_directory = Path("./")
    with open(output_directory / 'predictions.jsonl', 'w') as f:
        for entry in output:
            f.write(json.dumps(entry) + '\n')
