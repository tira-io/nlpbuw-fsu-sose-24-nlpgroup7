import json
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from tira.rest_api_client import Client
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification

def load_data(text_data, label_data):
    texts = text_data.to_dict(orient='records')
    labels = label_data.to_dict(orient='records')
    data = []
    for text, label in zip(texts, labels):
        data.append({
            'id': text['id'],
            'sentence': text['sentence'].split(),
            'labels': label['tags']
        })
    return data

def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples['sentence'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if __name__ == "__main__":
    tira = Client()

    # Load training and validation data
    # text_train = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-train-20240612-training")
    # targets_train = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-train-20240612-training")
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")

    # Process data
    # train_data = load_data(text_train, targets_train)
    val_data = load_data(text_validation, targets_validation)

    # Load pre-trained model and tokenizer
    model_name = "bert-base-cased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=9)  # Adjust num_labels as needed

    # Create datasets
    # train_dataset = Dataset.from_pandas(pd.DataFrame(train_data)).map(tokenize_and_align_labels, batched=True)
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data)).map(tokenize_and_align_labels, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize Trainer
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=val_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model('./saved_model')
    tokenizer.save_pretrained('./saved_model')
