import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Fish Species Classification",
    page_icon="🐟",
    layout="centered"
)

st.title("🐟 Fish Species Classification")
st.write("Upload a fish image and the model will predict its species.")


# -----------------------------
# Project paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

MODEL_PATH = BASE_DIR / "models" / "best_fish_classifier.h5"
METADATA_PATH = BASE_DIR / "best_model_metadata.csv"


# -----------------------------
# Class labels
# Must match training folder order
# -----------------------------
CLASS_NAMES = [
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


# -----------------------------
# Load model metadata
# -----------------------------
@st.cache_data
def load_best_model_name(metadata_path):
    metadata_df = pd.read_csv(metadata_path)
    return metadata_df.loc[0, "best_model"]


# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_trained_model(model_path):
    return load_model(model_path)


# -----------------------------
# Select preprocessing function
# -----------------------------
def get_preprocess_function(model_name):
    preprocess_map = {
        "MobileNetV2": mobilenet_preprocess,
        "ResNet50": resnet_preprocess,
        "VGG16": vgg16_preprocess,
        "InceptionV3": inception_preprocess,
        "EfficientNetB0": None
    }

    if model_name not in preprocess_map:
        raise ValueError(f"Unsupported model name: {model_name}")

    return preprocess_map[model_name]


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(uploaded_image, preprocess_fn):
    image = uploaded_image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    if preprocess_fn is not None:
        img_array = preprocess_fn(img_array)

    return image, img_array


# -----------------------------
# Prediction function
# -----------------------------
def predict_image(model, img_array, class_names, top_k=3):
    predictions = model.predict(img_array, verbose=0)[0]

    top_indices = predictions.argsort()[-top_k:][::-1]

    top_predictions = [
        {
            "class_name": class_names[index],
            "confidence": float(predictions[index])
        }
        for index in top_indices
    ]

    return top_predictions


# -----------------------------
# Load required assets
# -----------------------------
if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not METADATA_PATH.exists():
    st.error(f"Metadata file not found: {METADATA_PATH}")
    st.stop()

best_model_name = load_best_model_name(METADATA_PATH)
model = load_trained_model(MODEL_PATH)
preprocess_fn = get_preprocess_function(best_model_name)

st.caption(f"Current best model: {best_model_name}")


# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a fish image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction output
# -----------------------------
if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)

    display_image, img_array = preprocess_image(
        uploaded_image,
        preprocess_fn
    )

    st.image(
        display_image,
        caption="Uploaded Image",
        use_container_width=True
    )

    top_predictions = predict_image(
        model,
        img_array,
        CLASS_NAMES,
        top_k=3
    )

    best_prediction = top_predictions[0]

    st.subheader("Prediction")
    st.write("**Fish Species:**", best_prediction["class_name"])
    st.write("**Confidence:**", f"{best_prediction['confidence'] * 100:.2f}%")

    st.subheader("Top 3 Predictions")

    for prediction in top_predictions:
        st.write(
            f"{prediction['class_name']}: "
            f"{prediction['confidence'] * 100:.2f}%"
        )