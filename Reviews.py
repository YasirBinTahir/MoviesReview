import streamlit as st
import tensorflow as tf
import joblib # Changed from pickle to joblib, or keep pickle if you prefer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os # To check if files exist

# Inject CSS for background image and styled container
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    h1 {
      color: #2E86C1 !important;
    }
    p {
      color: green !important;
    }
    div[data-testid="stForm"] {
        background-color: rgba(0, 0, 0, 0.7); /* Slightly transparent black */
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
    }
    .form_submit_button {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define paths for your model and tokenizer
# These paths should match where you save them in your Colab training script
MODEL_PATH = 'final_sentiment_model.keras' # Make sure this matches the saved file name
TOKENIZER_PATH = 'tokenizer.pkl' # Make sure this matches the saved file name

# Load model and tokenizer
@st.cache_resource
def load_sentiment_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure it's in the same directory as your Streamlit app.")
        return None
    try:
        # Load the model directly
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None

@st.cache_data
def load_sentiment_tokenizer():
    if not os.path.exists(TOKENIZER_PATH):
        st.error(f"Tokenizer file not found at {TOKENIZER_PATH}. Please ensure it's in the same directory as your Streamlit app.")
        return None
    try:
        with open(TOKENIZER_PATH, 'rb') as f:
            return joblib.load(f) # Changed to joblib.load
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

model = load_sentiment_model()
tokenizer = load_sentiment_tokenizer()

# Make sure MAX_LEN is consistent with training
MAX_SEQUENCE_LENGTH = 100 # This must match MAX_LEN from your training script!

# Begin form
with st.form("sentiment_form"): # Renamed form ID for clarity
    st.title("🎬 Movie Review Sentiment Analyzer") # Adjusted title
    st.write("Enter a movie review to predict its sentiment (Positive or Negative).")

    review = st.text_area("Movie Review", height=100)

    # ✅ Submit button required for forms
    submit = st.form_submit_button("Predict Sentiment")

    if submit:
        if model is None or tokenizer is None:
            st.error("Model or tokenizer could not be loaded. Please check logs.")
        elif review:
            # Preprocess the input review
            sequence = tokenizer.texts_to_sequences([review])
            padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

            # Make prediction
            prediction = model.predict(padded)[0][0]
            sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"

            st.markdown(f"**Predicted Sentiment:** {sentiment}")
            st.markdown(f"*(Confidence: {prediction:.2f})*") # Show confidence for more insight
        else:
            st.warning("Please enter a movie review to analyze.")
