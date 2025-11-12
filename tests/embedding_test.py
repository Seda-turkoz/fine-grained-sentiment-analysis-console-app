from transformers import AutoTokenizer, AutoModel
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Example input
text = "I feel anxious but also hopeful."

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")

# Extract embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # sentence embedding

print("✅ Embedding vector shape:", embeddings.shape)
print("🧠 Embedding sample:", embeddings[0][:10])  # print first 10 values
