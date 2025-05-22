import streamlit as st
import tensorflow as tf
import requests
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from typing import Optional

# ----------------------------
# Configuration
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_H5_PATH = os.path.join(BASE_DIR, "movie_review_sentiment_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pickle")
MAX_LEN = 200  # Must match training

# ----------------------------
# Load Resources (cached)
# ----------------------------

# Google Drive file ID (you extract it from the share link)
FILE_ID = "YOUR_FILE_ID_HERE"  # ← Change this!
MODEL_PATH = "movie_review_sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"
MAX_LEN = 200

# Direct download helper
def download_from_gdrive(file_id, dest_path):
    URL = "https://drive.google.com/file/d/1qYxA1WodMtuCTaf6ZVfznW78-G7h6efz/view?usp=sharing"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    save_response_content(response, dest_path)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_model_and_tokenizer():
    # Download model if not already present
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model...")
        download_from_gdrive(FILE_ID, MODEL_PATH)

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)

    return model, tokenizer, MAX_LEN

# ----------------------------
# Sentiment Prediction
# ----------------------------

def predict_sentiment(text: str, model: tf.keras.Model, tokenizer: Tokenizer, max_len: int) -> str:
    if not model or not tokenizer:
        return "Model or tokenizer not loaded."

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

    try:
        prediction = model.predict(padded)[0][0]
        return "😊 Positive" if prediction >= 0.5 else "😞 Negative"
    except Exception as e:
        return f"Prediction error: {e}"

# ----------------------------
# Streamlit UI
# ----------------------------

def main():
    st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")

    # 🎬 Background styling
    st.markdown(
        """
        <style>
        .stApp {
            background-image: linear-gradient(
                rgba(0, 0, 0, 0.65),
                rgba(0, 0, 0, 0.65)
            ), url("https://images.unsplash.com/photo-1598899134739-24c46f58b8e0?auto=format&fit=crop&w=1470&q=80");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: white;
        }

        h1, .stMarkdown, .stTextInput label, .stTextArea label {
            color: white !important;
        }

        textarea {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: black !important;
        }

        .stButton>button {
            background-color: #e50914;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🎬 Movie Review Sentiment Analyzer")
    st.write("Type a movie review below and click **Analyze Sentiment** to see if it's positive or negative.")

    # Load model and tokenizer
    model, tokenizer, max_len = load_resources()
    if model is None or tokenizer is None:
        st.stop()

    # Input and prediction
    user_input = st.text_area("✍️ Enter your movie review:", "The movie was absolutely fantastic!")

    if st.button("Analyze Sentiment"):
        sentiment = predict_sentiment(user_input, model, tokenizer, max_len)
        st.success(f"**Result:** {sentiment}")

if __name__ == "__main__":
    main()
