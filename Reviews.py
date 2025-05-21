import streamlit as st
import tensorflow as tf
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

@st.cache_resource
def load_resources() -> tuple[Optional[tf.keras.Model], Optional[Tokenizer], int]:
    try:
        with open(TOKENIZER_PATH, "rb") as handle:
            tokenizer = pickle.load(handle)
    except Exception as e:
        st.error(f"❌ Could not load tokenizer: {e}")
        return None, None, None

    try:
        model = tf.keras.models.load_model(MODEL_H5_PATH)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None, None, None

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
