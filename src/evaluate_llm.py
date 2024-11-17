from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from evaluate import load
import torch
import time
import numpy as np

# Load Evaluation Metrics
bleu = load("sacrebleu")
rouge = load("rouge")
accuracy = load("accuracy")

# Initialize Model
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Compute Perplexity
def compute_perplexity(model, tokenizer, text_list):
    encodings = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss.item()
    perplexity = torch.exp(torch.tensor(loss))
    return perplexity.item()

# Compute BLEU
def compute_bleu(predictions, references):
    return bleu.compute(predictions=predictions, references=references)

# Compute ROUGE
def compute_rouge(predictions, references):
    return rouge.compute(predictions=predictions, references=references)

# Benchmark LLM
def benchmark_model(model_name, test_dataset, max_samples=100):
    tokenizer, model = load_model(model_name)
    pipeline_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    perplexities, predictions, references, response_times = [], [], [], []
    for i, sample in enumerate(test_dataset):
        if i >= max_samples:
            break
        input_text = sample['input']
        reference_text = sample['output']

        # Measure Response Time
        start_time = time.time()
        generated_text = pipeline_gen(input_text, max_length=50, num_return_sequences=1)[0]['generated_text']
        end_time = time.time()
        response_times.append(end_time - start_time)
        
        # Compute Perplexity
        perplexity = compute_perplexity(model, tokenizer, [input_text])
        perplexities.append(perplexity)

        predictions.append(generated_text)
        references.append(reference_text)

    # Metrics Aggregation
    avg_perplexity = np.mean(perplexities)
    avg_response_time = np.mean(response_times)
    bleu_score = compute_bleu(predictions, references)
    rouge_score = compute_rouge(predictions, references)

    results = {
        "avg_perplexity": avg_perplexity,
        "avg_response_time": avg_response_time,
        "bleu": bleu_score,
        "rouge": rouge_score
    }
    return results

# Test the Framework
if __name__ == "__main__":
    model_name = "gpt2"
    dataset = load_dataset("xsum", split="test[:1%]")  # Example dataset
    test_data = [{"input": ex["document"], "output": ex["summary"]} for ex in dataset]
    
    results = benchmark_model(model_name, test_data, max_samples=10)
    print("Benchmark Results:", results)

