from datasets import load_dataset
import torch
import numpy as np
import os
# Emotion list (id → label mapping) (27 + neutral)

emotion_list = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]


def generate_labels_if_needed():
    """
    Check if 'train_labels.pt' exists. If not, generate and save labels.
    """
    os.makedirs("trained_models", exist_ok=True)  # 📁 Klasör yoksa oluştur

    if os.path.exists("trained_models/train_labels.pt"):
        print("📦 Labels already exist. Skipping generation.")
        return

    print("🔄 Loading dataset for label generation...")
    dataset = load_dataset(
        "json", data_files="archive/data/train_preprocessed.jsonl", split="train")

    all_labels = []
    for example in dataset:
        label_vector = np.zeros(len(emotion_list))
        for label in example["labels"]:
            if label in emotion_list:
                idx = emotion_list.index(label)
                label_vector[idx] = 1
        all_labels.append(label_vector)

    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
    torch.save(labels_tensor, "trained_models/train_labels.pt")
    print("✅ Labels saved to 'trained_models/train_labels.pt' with shape:",
          labels_tensor.shape)


def get_label_names():
    return [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
    ]


id2emotion = {i: label for i, label in enumerate(emotion_list)}
