import pandas as pd
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from rouge_score import rouge_scorer
import torch

def generate_summary(model, tokenizer, text, device, max_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(score[key].fmeasure)
    avg_scores = {key: sum(values) / len(values) for key, values in scores.items()}
    return avg_scores

if __name__ == "__main__":
    # Check if a GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    labels_val = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    # Load the fine-tuned model and tokenizer
    model_path = "./results_bigger_model_tuning/trained_model"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)  # Move the model to the device

    # Generate summaries for each story
    df["summary"] = df["story"].apply(lambda x: generate_summary(model, tokenizer, x, device))
    
    # Calculate ROUGE scores (optional, can be commented out)
    reference_summaries = labels_val["summary"]  # assuming the reference summaries are in this column
    generated_summaries = df["summary"]
    rouge_scores = compute_rouge(generated_summaries, reference_summaries)
    print("ROUGE scores:", rouge_scores)

    # Log final evaluation results
    with open("./results_bigger_model_tuning/results_compute_rouge/metrics_log_cuda.txt", "a") as writer:
        writer.write(f"Final Evaluation results: {rouge_scores}\n")

    # Save the predictions
    df = df.drop(columns=["story"]).reset_index()
    output_directory = get_output_directory(str(Path(__file__).parent))
    # df.to_json(
    #     Path(output_directory) / "results_bigger_model/results_compute_rouge/predictions.jsonl", orient="records", lines=True
    # )

    df.to_json(
        "./results_bigger_model_tuning/results_compute_rouge/predictions.jsonl", orient="records", lines=True
    )
