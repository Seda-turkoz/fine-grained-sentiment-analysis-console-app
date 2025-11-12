import torch
import numpy as np
import joblib
from utils.label_map import id2emotion
import os
from ai_based.embedding_pipeline import get_embedding_from_text, generate_embeddings_if_needed
from utils.save_labels import generate_labels_if_needed
from ai_based.logistic_regression.train_logistic_regression import train_logistic_regression_model
from utils.save_labels import generate_labels_if_needed
from ai_based.visualization_utils import plot_emotion_pie


def ensure_all_artifacts():
    model_path = "trained_models/logisticregression_model.pkl"

    if not os.path.exists("trained_models/train_embeddings.pt"):
        print("📥 Embedding not found. Generating...")
        generate_embeddings_if_needed()

    if not os.path.exists("trained_models/train_labels.pt"):
        print("📥 Label file not found. Generating...")
        generate_labels_if_needed()

    if not os.path.exists(model_path):
        print("🧠 Logistic Regression model not found. Training...")
        train_logistic_regression_model()

    return joblib.load(model_path)


def predict_with_logistic(text: str):
    """
    Predict emotion(s) from raw text using DistilBERT embedding + Logistic Regression.
    """
    # 🔧 Gereken dosyaları kontrol et → gerekiyorsa üret
    model = ensure_all_artifacts()

    # 🔍 Embedding üret
    embedding = get_embedding_from_text(text)

    # Step 2: Ensure it's in correct format
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.numpy()
    embedding = np.expand_dims(embedding, axis=0)

    # Step 3: Predict
    prediction = model .predict(embedding)[0]

    # Step 4: Convert prediction indices to emotion labels
    predicted_emotions = [id2emotion[i]
                          for i, val in enumerate(prediction) if val == 1]

    # Predict with probability
    proba = model.predict_proba(embedding)[0]
    # Get top N (e.g., top 3) emotions with highest probabilities
    top_indices = np.argsort(proba)[::-1][:3]
    top_emotions = [id2emotion[i] for i in top_indices]
    top_scores = [proba[i] for i in top_indices]

# Draw pie chart
    plot_emotion_pie(top_emotions, top_scores,
                     model_name="Logistic Regression")
    return predicted_emotions
