import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Define device (uses CUDA GPU if available, otherwise defaults to CPU)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Running on device: {device}")

# 2. Download and load the pre-trained FinBERT tokenizer and model
model_name = "ProsusAI/finbert"
print("[INFO] Loading FinBERT model and tokenizer from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# 3. Define sample test headlines to evaluate
test_headlines = [
    "Apple quarterly profits beat Wall Street estimates by 25%",
    "Tesla faces major factory shutdown and supply chain disruptions",
    "Markets open flat ahead of Federal Reserve interest rate decision"
]

# 4. Tokenization: Convert text into numerical tensors
tokens = tokenizer(test_headlines, padding=True, truncation=True, return_tensors="pt").to(device)

# 5. Model Inference: Forward pass through neural network
with torch.no_grad():
    outputs = model(**tokens)
    # Convert raw logits to probability scores using softmax
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

# FinBERT output class indices: 0 -> positive, 1 -> negative, 2 -> neutral
labels = ["Positive", "Negative", "Neutral"]

print("\n--- Sentiment Classification Results ---")
for idx, headline in enumerate(test_headlines):
    scores = probabilities[idx]
    best_class_idx = torch.argmax(scores).item()
    best_label = labels[best_class_idx]
    confidence = scores[best_class_idx].item()
    
    print(f"\nHeadline: \"{headline}\"")
    print(f"Predicted Sentiment: {best_label}")
    print(f"Confidence Score   : {confidence:.2%}")
    print(f"Full Probabilities : Positive={scores[0]:.2%}, Negative={scores[1]:.2%}, Neutral={scores[2]:.2%}")