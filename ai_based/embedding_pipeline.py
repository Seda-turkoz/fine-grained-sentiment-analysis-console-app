from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()  # Disable gradient calculation


def get_embedding_from_text(text: str):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


def generate_embeddings_if_needed():
    """
    Check if 'train_embeddings.pt' exists. If not, generate and save embeddings.
    """
    os.makedirs("trained_models", exist_ok=True)  # 📁 Klasör yoksa oluştur
    if os.path.exists("trained_models/train_embeddings.pt"):
        print("📦 Embeddings already exist. Skipping generation.")
        return

    print("🔄 Loading dataset...")
    dataset = load_dataset(
        "json", data_files="archive/data/train_preprocessed.jsonl", split="train")

    embeddings = []
    for example in tqdm(dataset, desc="Generating embeddings"):
        embedding = get_embedding_from_text(example["text"])
        embeddings.append(embedding)

    torch.save(embeddings, "trained_models/train_embeddings.pt")
    print("✅ Embeddings saved to 'trained_models/train_embeddings.pt'")
