# app.py
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and preprocessors
@st.cache_resource
def load_artifacts():
    model = load_model("gender_name_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_artifacts()

st.title("👤 Tamil Name Gender Prediction App")

# User input
name_input = st.text_input("Enter a Tamil name:")

if st.button("Predict"):
    if name_input.strip() == "":
        st.warning("Please enter a name.")
    else:
        # Preprocess input
        seq = tokenizer.texts_to_sequences([name_input])
        max_len = model.input_shape[1]
        seq_padded = pad_sequences(seq, maxlen=max_len, padding="post")

        # Predict
        prob = model.predict(seq_padded)[0][0]
        pred_class = (prob > 0.5).astype(int)
        gender = label_encoder.inverse_transform([pred_class])[0]

        st.success(f"Predicted Gender: **{gender}** (Confidence: {prob:.2f})")
