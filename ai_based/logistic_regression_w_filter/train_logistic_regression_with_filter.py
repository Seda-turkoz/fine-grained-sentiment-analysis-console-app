import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
import spacy
import os
import joblib
import spacy
import coreferee


# Load embeddings and labels
try:
    embeddings = torch.load("trained_models/train_embeddings.pt")

    # Eğer her bir embedding ayrı bir tensor ise
    if isinstance(embeddings, list) and isinstance(embeddings[0], torch.Tensor):
        X_train = np.vstack([emb.numpy() for emb in embeddings])
    # Eğer zaten hepsi tek bir tensor'sa
    elif isinstance(embeddings, torch.Tensor):
        X_train = embeddings.numpy()
    else:
        raise ValueError("Unexpected embedding format.")

    y_train = np.array(torch.load("trained_models/train_labels.pt"))

except Exception as e:
    print("🚨 Error Accured:", e)
    exit()


# MODEL_PATH = "logisticregression_model.pkl"
MODEL_PATH = os.path.join(os.getcwd(), "logisticregression_model.pkl")
print("📁 Model saved to:", os.path.abspath(MODEL_PATH))
if os.path.exists(MODEL_PATH):
    print("📦 Loading existing model...")
    clf = joblib.load(MODEL_PATH)
else:
    print("⚙️ Training new model...")
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_PATH)
    print("✅ Model saved as", MODEL_PATH)
