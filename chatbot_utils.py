"""Data loading, training and prediction helpers for the intent chatbot.

Kept free of any Streamlit import so it can be tested and reused on its own.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FALLBACK_TAG = "fallback"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


def load_intents(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(intents_data: dict) -> tuple[list[str], list[str]]:
    """Flatten intents.json into (pattern_text, tag) pairs."""
    texts, tags = [], []
    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(pattern)
            tags.append(intent["tag"])
    return texts, tags


def make_pipeline(seed: int) -> Pipeline:
    # C=6 sharpens the predicted probabilities (a small dataset spread over many
    # classes otherwise keeps every confidence low even when the top guess is right).
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, C=6.0, random_state=seed)),
        ]
    )


def _safe_split(texts, tags, test_size, seed):
    """Stratified split, falling back to a plain split if a class is too small."""
    try:
        return train_test_split(
            texts, tags, test_size=test_size, random_state=seed, stratify=tags
        )
    except ValueError:
        return train_test_split(texts, tags, test_size=test_size, random_state=seed)


def train_and_evaluate(intents_data: dict, test_size: float = 0.2, seed: int = 42) -> dict:
    """Train on a split to measure accuracy, then fit a final model on all data.

    Returns everything the app needs: the deployable pipeline, held-out
    accuracy, a per-tag report, and the full label list.
    """
    texts, tags = build_dataset(intents_data)
    labels = sorted(set(tags))

    X_train, X_test, y_train, y_test = _safe_split(texts, tags, test_size, seed)

    eval_pipe = make_pipeline(seed)
    eval_pipe.fit(X_train, y_train)
    y_pred = eval_pipe.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    # Final model deployed in the app is trained on the full pattern set,
    # so the chatbot itself has seen every example, not just the training split.
    final_pipe = make_pipeline(seed)
    final_pipe.fit(texts, tags)

    return {
        "pipeline": final_pipe,
        "accuracy": accuracy,
        "report": report,
        "labels": labels,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_total": len(texts),
    }


def predict_tag(pipeline: Pipeline, text: str) -> tuple[str, float]:
    """Predict the intent tag for a message and return (tag, confidence)."""
    probs = pipeline.predict_proba([text])[0]
    idx = int(np.argmax(probs))
    tag = pipeline.classes_[idx]
    return str(tag), float(probs[idx])


def get_response(
    intents_data: dict,
    tag: str,
    confidence: float,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    rng: random.Random | None = None,
) -> str:
    """Pick a reply for the predicted tag, or a fallback line if confidence is low."""
    rng = rng or random
    if confidence < threshold:
        return rng.choice(intents_data["fallback_responses"])
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return rng.choice(intent["responses"])
    return rng.choice(intents_data["fallback_responses"])


def chat_reply(
    pipeline: Pipeline,
    intents_data: dict,
    text: str,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    rng: random.Random | None = None,
) -> dict:
    """One full turn: predict the tag and produce the reply text."""
    text = text.strip()
    if not text:
        return {"tag": FALLBACK_TAG, "confidence": 0.0, "response": "Say something and I'll reply!"}
    tag, confidence = predict_tag(pipeline, text)
    response = get_response(intents_data, tag, confidence, threshold, rng)
    return {"tag": tag, "confidence": confidence, "response": response}
