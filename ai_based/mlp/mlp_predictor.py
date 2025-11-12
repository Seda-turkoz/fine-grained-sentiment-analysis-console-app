import os
import joblib
import torch
import numpy as np
from ai_based.embedding_pipeline import generate_embeddings_if_needed, get_embedding_from_text
from utils.save_labels import generate_labels_if_needed
from ai_based.mlp.train_mlp_classifier import train_mlp_classifier_model
from utils.save_labels import id2emotion
from ai_based.visualization_utils import plot_emotion_pie


def ensure_all_artifacts_for_mlp():

    model_path = "trained_models/mlp_classifier_model.pkl"

    """
    Check for embeddings, labels, and trained MLP model. Generate or train if needed.
    """
    if not os.path.exists("trained_models/train_embeddings.pt"):
        print("📥 Embedding not found. Generating...")
        generate_embeddings_if_needed()

    if not os.path.exists("trained_models/train_labels.pt"):
        print("📥 Label file not found. Generating...")
        generate_labels_if_needed()

    if not os.path.exists(model_path):
        print("🧠 MLP model not found. Training...")
        train_mlp_classifier_model()

    return joblib.load(model_path)


def predict_with_mlp(text: str):
    """
    Predict emotion(s) from raw text using DistilBERT embedding + MLP classifier.
    """
    # 1. Ensure all components are ready
    model = ensure_all_artifacts_for_mlp()

    # 2. Generate embedding from input text
    embedding = get_embedding_from_text(text)

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.numpy()
    embedding = np.expand_dims(embedding, axis=0)

    # 3. Predict
    prediction = model.predict(embedding)[0]

    # 4. Convert indices to emotion labels
    predicted_emotions = [id2emotion[i]
                          for i, val in enumerate(prediction) if val == 1]
    # Predict with probability (görsel çizimi için)
    try:
        proba = model.predict_proba(embedding)[0]
        top_indices = np.argsort(proba)[::-1][:3]
        top_emotions = [id2emotion[i] for i in top_indices]
        top_scores = [proba[i] for i in top_indices]
    except AttributeError:
        # Eğer MLPClassifier predict_proba desteklemiyorsa fallback
        top_emotions = predicted_emotions
        top_scores = [1 / len(predicted_emotions)] * len(predicted_emotions)

    # 🎯 Pie chart çiz
    plot_emotion_pie(top_emotions, top_scores, model_name="MLP Classifier")
    return predicted_emotions
