import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib
import os


def train_logistic_regression_model():
    raw_embeddings = torch.load(
        "trained_models/train_embeddings.pt")  # list of tensors
    X_train = np.array([emb.numpy() for emb in raw_embeddings])
    y_train = torch.load("trained_models/train_labels.pt")
    y_train = np.array(y_train)

    model_path = "trained_models/logisticregression_model.pkl"

    if os.path.exists(model_path):
        print("📦 Model already trained. Skipping training.")
        return joblib.load(model_path)
    else:
        print("🚀 No model found. Training from scratch...")
        model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        model.fit(X_train, y_train)
        os.makedirs("trained_models", exist_ok=True)
        joblib.dump(model, model_path)
        print("✅ Model trained and saved to", model_path)
        return model
