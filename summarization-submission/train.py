import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# Load the data
tira = Client()

text_train = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "summarization-train-20240530-training"
).set_index("id")
labels_train = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "summarization-train-20240530-training"
).set_index("id")

text_val = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
).set_index("id")
labels_val = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
).set_index("id")

# Prepare the data for training
def prepare_data(text, labels):
    data = {'input_text': text['story'].tolist(), 'summary': labels['summary'].tolist()}
    return pd.DataFrame(data)

train_data = prepare_data(text_train, labels_train)
val_data = prepare_data(text_val, labels_val)

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

# Tokenizer and model initialization
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize the data
def preprocess_function(examples):
    inputs = [doc for doc in examples["input_text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=150, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=3,
    predict_with_generate=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
