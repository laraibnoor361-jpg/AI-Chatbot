"""
Simple AI Chatbot - Streamlit App
---------------------------------
A rule/ML based chatbot. It reads intents.json (patterns + responses),
trains a TF-IDF + Logistic Regression classifier to guess the intent
of whatever the user types, and replies with a matching response.

Run locally:
    streamlit run app.py
"""

import random
from pathlib import Path

import pandas as pd
import streamlit as st

from chatbot_utils import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    chat_reply,
    load_intents,
    train_and_evaluate,
)

INTENTS_FILE = Path(__file__).parent / "intents.json"

st.set_page_config(page_title="Simple AI Chatbot", page_icon="🤖")

st.title("🤖 Simple AI Chatbot")
st.write(
    "This is a small chatbot made for a machine learning class project. "
    "It uses TF-IDF + Logistic Regression to guess what you mean and reply. "
    "Try greetings, asking for a joke, a study tip, or just say bye."
)


@st.cache_data(show_spinner=False)
def get_intents():
    return load_intents(INTENTS_FILE)


@st.cache_resource(show_spinner=False)
def get_model(test_size, seed):
    intents_data = get_intents()
    return train_and_evaluate(intents_data, test_size=test_size, seed=seed)


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20, step=5) / 100
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
threshold = st.sidebar.slider(
    "Confidence threshold",
    0.05, 0.60, DEFAULT_CONFIDENCE_THRESHOLD, step=0.05,
    help="If the model isn't at least this confident, it gives a fallback reply instead of guessing.",
)

intents_data = get_intents()
model = get_model(test_size, seed)

st.sidebar.subheader("Model info")
st.sidebar.write(f"Training examples: {model['n_train']}")
st.sidebar.write(f"Test examples: {model['n_test']}")
st.sidebar.write(f"Test accuracy: {model['accuracy']:.2%}")
st.sidebar.caption(
    "Accuracy is measured on patterns the model did not train on. "
    "The chatbot itself is then retrained on all patterns before replying to you."
)

with st.sidebar.expander("What intents (tags) does it know?"):
    tags = [intent["tag"] for intent in intents_data["intents"]]
    counts = pd.Series(
        [len(intent["patterns"]) for intent in intents_data["intents"]], index=tags
    )
    st.dataframe(counts.rename("example patterns"))

if st.sidebar.button("Clear chat"):
    st.session_state["messages"] = []
    st.rerun()

# ---------------- Chat state ----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm StudyBot. Ask me something or say hi."}
    ]
if "rng" not in st.session_state:
    st.session_state["rng"] = random.Random(seed)

# show chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# chat input
user_text = st.chat_input("Type a message...")
if user_text:
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    result = chat_reply(
        model["pipeline"], intents_data, user_text,
        threshold=threshold, rng=st.session_state["rng"],
    )
    reply = result["response"]

    with st.chat_message("assistant"):
        st.write(reply)
        st.caption(f"(predicted intent: {result['tag']}, confidence: {result['confidence']:.2f})")

    st.session_state["messages"].append({"role": "assistant", "content": reply})
