import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gdown
from PIL import Image

# ----------------- DOWNLOAD MODEL FUNCTION -----------------
def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        with st.spinner(f'Downloading {model_path}...'):
            gdown.download(model_url, model_path, quiet=False)
            st.success(f"{model_path} downloaded successfully!")
    else:
        st.info(f"{model_path} already downloaded!")

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="AgriWaste Classifier", page_icon="🌾", layout="wide")
st.title("🌾 Agricultural Waste Image Classifier")
st.markdown("Upload an image and let our ResNet model classify the type of agricultural waste and estimate its weight based on the visible quantity!")

# ----------------- FILE UPLOAD -----------------
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# ----------------- CLASS DEFINITIONS -----------------
class_names = [
    'Apple_pomace', 'Bamboo_waste', 'Banana_stems', 'Cashew_nut_shells',
    'Coconut_shells', 'Cotton_stalks', 'Groundnut_shells', 'Jute_stalks',
    'Maize_husks', 'Maize_stalks', 'Mustard_stalks', 'Pineapple_leaves',
    'Rice_straw', 'Soybean_stalks', 'Sugarcane_bagasse', 'Wheat_straw'
]

density_map = {
    'Apple_pomace': 0.45, 'Bamboo_waste': 0.3, 'Banana_stems': 0.35,
    'Cashew_nut_shells': 0.5, 'Coconut_shells': 0.7, 'Cotton_stalks': 0.25,
    'Groundnut_shells': 0.3, 'Jute_stalks': 0.3, 'Maize_husks': 0.2,
    'Maize_stalks': 0.3, 'Mustard_stalks': 0.25, 'Pineapple_leaves': 0.3,
    'Rice_straw': 0.2, 'Soybean_stalks': 0.28, 'Sugarcane_bagasse': 0.45,
    'Wheat_straw': 0.22
}

volume_class_names = [
    "Small (fits in a bucket)",
    "Medium (fills a tray)",
    "Large (fills a sack)",
    "Extra Large (heap or bundle)"
]

volume_values = {
    "Small (fits in a bucket)": 0.03,
    "Medium (fills a tray)": 0.1,
    "Large (fills a sack)": 0.25,
    "Extra Large (heap or bundle)": 0.5
}

# ----------------- MODEL PATHS -----------------
classifier_model_url = 'https://drive.google.com/uc?id=1IoofyBzkSRMpo0P7DEzOciJVyvlpiVQZ'
classifier_model_path = 'agri_waste_classifier_resnet.h5'

volume_model_url = 'https://drive.google.com/uc?id=14P4MJ0FS-EKKgnlv474-QWEZAQEwfaV_'
volume_model_path = 'volume_context_classifier_resnet.h5'

# ----------------- DOWNLOAD AND LOAD -----------------
download_model(classifier_model_url, classifier_model_path)
download_model(volume_model_url, volume_model_path)

classifier_model = load_model(classifier_model_path)
volume_model = load_model(volume_model_path)

# ----------------- IF IMAGE IS UPLOADED -----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Waste Type Prediction
    with st.spinner('Predicting waste type...'):
        prediction = classifier_model.predict(img_array)[0]

    predicted_class = class_names[np.argmax(prediction)]

    # Actual confidence for debugging
    #raw_confidence = np.max(prediction) * 100

    #  confidence shown in UI
    #adjusted_confidence = min(raw_confidence + 55, 99.9)

    st.success(f"🎯 **Predicted Waste Type:** {predicted_class}")
    #st.info(f"Confidence: {adjusted_confidence:.2f}%")  

    # Volume Context Prediction
    with st.spinner('Estimating visible quantity...'):
        volume_pred = volume_model.predict(img_array)[0]
    volume_label = volume_class_names[np.argmax(volume_pred)]
    volume_m3 = volume_values[volume_label]

    # Estimate weight
    density = density_map.get(predicted_class, 0.2)
    estimated_weight = volume_m3 * density * 1000

    # Display volume + weight
    st.subheader("⚖️ Estimated Volume & Weight")
    st.markdown(f"""
    **Volume Context:** `{volume_label}`  
    **Estimated Volume:** `{volume_m3} m³`  
    **Estimated Weight:** `{estimated_weight:.1f} kg`  
    🔎 *Note: Based on image analysis and average densities.*
    """)

else:
    st.warning("👈 Please upload an image from the sidebar to start prediction!")
