import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
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
model_name = "t5-small"  # Use a smaller model to reduce memory usage
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize the data
def preprocess_function(examples):
    inputs = [doc for doc in examples["input_text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "summary"])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    save_strategy="steps",
    metric_for_best_model="rouge1",
    greater_is_better=True,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Further reduced batch size
    per_device_eval_batch_size=1,   # Further reduced batch size
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,  # Mixed precision training
    device="cuda",  # Utilize CUDA for training
)

# Define the metric computation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = [[label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Use ROUGE score for evaluation
    from datasets import load_metric
    rouge = load_metric("rouge")
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Extract the f1 score of the ROUGE metric
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
