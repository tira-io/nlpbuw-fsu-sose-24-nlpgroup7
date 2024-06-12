import pandas as pd
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from tqdm import tqdm

def generate_summary(model, tokenizer, text, max_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_in_batches(df, model, tokenizer, batch_size=10):
    summaries = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        batch_summaries = batch["story"].apply(lambda x: generate_summary(model, tokenizer, x))
        summaries.extend(batch_summaries)
    return summaries

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    # Load the fine-tuned model and tokenizer
    model_path = "/code/trained_model"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    # Generate summaries in batches and show progress
    batch_size = 10 
    df["summary"] = process_in_batches(df, model, tokenizer, batch_size)
    df = df.drop(columns=["story"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
