import os
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from ai_based.embedding_pipeline import generate_embeddings_if_needed, get_embedding_from_text
from utils.save_labels import generate_labels_if_needed
from ai_based.logistic_regression.train_logistic_regression import train_logistic_regression_model
from ai_based.visualization_utils import plot_emotion_pie
from utils.label_map import id2emotion
import re
import string
import spacy


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


_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def is_emotion_from_author_basic(text):
    """Kelime bazlı en basit kontrol."""
    text_lower = f" {text.lower()} "
    return any(f" {word} " in text_lower for word in ["i", "me", "my", "myself"])


def is_emotion_from_author_strict(text):
    nlp = get_nlp()
    doc = nlp(text)
    for sent in doc.sents:
        if any(tok.lemma_.lower() in ["i", "me", "my", "myself"] for tok in sent):
            return True
    return False


def is_emotion_from_author_v2(text):
    import coreferee   # required for registering 'coreferee' pipe

    nlp = get_nlp()

    if "coreferee" not in nlp.pipe_names:
        nlp.add_pipe("coreferee")

    """
    Coreference + fallback bazlı ana kontrol.
    1. Önce coreference analiz eder.
    2. Bulunamazsa strict → basic fallback zinciriyle kontrol eder.
    """
    try:
        doc = nlp(text)
        if not doc._.coref_chains or len(doc._.coref_chains) == 0:
            # print("🌀 Coref chains not found. Falling back.")
            return is_emotion_from_author_strict(text) or is_emotion_from_author_basic(text)

        for chain in doc._.coref_chains:
            referents = [mention.text.lower()
                         for mention in chain.mentions if hasattr(mention, "text")]
            if any(ref in referents for ref in ["i", "me", "my", "myself"]):
                return True

        # print("🚫 First-person referent not found in chains. Fallback active.")
        return is_emotion_from_author_basic(text) or is_emotion_from_author_strict(text)

    except Exception as e:
        print("⚠️ Coreference resolution failed:", str(e))
        return is_emotion_from_author_basic(text) or is_emotion_from_author_strict(text)


def predict_with_logistic_attribution(text: str):
    """
    Predict emotion(s) using Logistic Regression + Attribution Filter (1st-person only).
    """
    if not is_emotion_from_author_v2(text):
        # print("🛑 Emotion not attributed to the author. Skipping prediction.")
        return ["[Filtered: Not Author's Emotion]"]

    model = ensure_all_artifacts()
    embedding = get_embedding_from_text(text)

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.numpy()
    embedding = np.expand_dims(embedding, axis=0)

    prediction = model.predict(embedding)[0]
    predicted_emotions = [id2emotion[i]
                          for i, val in enumerate(prediction) if val == 1]

    return predicted_emotions
