import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Page title
st.title("🐟 Fish Species Classification")

st.write("Upload a fish image and the model will predict the species.")

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_fish_classifier.h5")
model = load_model(model_path)

# Class names (must match training folders)
class_names = [
    "animal fish",
    "animal fish bass",
    "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream",
    "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout"
]

# Image uploader
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224,224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader("Prediction")

    st.write("Fish Species:", predicted_class)
    st.write("Confidence:", round(confidence * 100, 2), "%")