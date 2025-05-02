import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gdown
from PIL import Image
import matplotlib.pyplot as plt

# Function to download the model if not present locally
def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        with st.spinner('Downloading the model...'):
            gdown.download(model_url, model_path, quiet=False)
            st.success("Model downloaded successfully!")
    else:
        st.info("Model already downloaded!")

# Load model
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Set page config
st.set_page_config(page_title="AgriWaste Classifier", page_icon="🌾", layout="wide")

# Title
st.title("🌾 Agricultural Waste Image Classifier")
st.markdown("Upload an image and let our ResNet model classify the type of agricultural waste!")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Class names
class_names = [
    'Apple_pomace',
    'Bamboo_waste',
    'Banana_stems',
    'Cashew_nut_shells',
    'Coconut_shells',
    'Cotton_stalks',
    'Groundnut_shells',
    'Jute_stalks',
    'Maize_husks',
    'Maize_stalks',
    'Mustard_stalks',
    'Pineapple_leaves',
    'Rice_straw',
    'Soybean_stalks',
    'Sugarcane_bagasse',
    'Wheat_straw'
]

# Densities in g/cm³
density_map = {
    'Apple_pomace': 0.45,
    'Bamboo_waste': 0.3,
    'Banana_stems': 0.35,
    'Cashew_nut_shells': 0.5,
    'Coconut_shells': 0.7,
    'Cotton_stalks': 0.25,
    'Groundnut_shells': 0.3,
    'Jute_stalks': 0.3,
    'Maize_husks': 0.2,
    'Maize_stalks': 0.3,
    'Mustard_stalks': 0.25,
    'Pineapple_leaves': 0.3,
    'Rice_straw': 0.2,
    'Soybean_stalks': 0.28,
    'Sugarcane_bagasse': 0.45,
    'Wheat_straw': 0.22
}

# Volume options in m³
volume_options = {
    "Small (fits in a bucket)": 0.03,
    "Medium (fills a tray)": 0.1,
    "Large (fills a sack)": 0.25,
    "Extra Large (heap or bundle)": 0.5
}

# Define model path and Google Drive URL for the model
model_url = 'https://drive.google.com/uc?id=1IoofyBzkSRMpo0P7DEzOciJVyvlpiVQZ'
model_path = 'agri_waste_classifier_resnet.h5'

# Download and load model
download_model(model_url, model_path)
model = load_model(model_path)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner('Predicting...'):
        prediction = model.predict(img_array)[0]

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🎯 **Predicted Class:** {predicted_class}")
    st.info(f"🔍 **Confidence:** {confidence:.2f}%")

    # Bar chart
    st.subheader("📊 Prediction Probabilities:")
    prediction_df = pd.DataFrame({
        'Class': class_names,
        'Confidence': prediction * 100
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(prediction_df['Class'], prediction_df['Confidence'], color='mediumseagreen')
    ax.set_xlabel('Confidence (%)')
    ax.set_xlim(0, 100)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # --- NEW: Estimate Weight ---
    st.subheader("⚖️ Estimated Weight Calculator")

    selected_volume_label = st.selectbox("Select visible pile size:", list(volume_options.keys()))
    volume_m3 = volume_options[selected_volume_label]

    density = density_map.get(predicted_class, 0.2)
    estimated_weight = volume_m3 * density * 1000  # in kg

    st.markdown(f"""
    **Waste Type:** `{predicted_class}`  
    **Estimated Volume:** `{volume_m3} m³`  
    **Estimated Weight:** `{estimated_weight:.1f} kg`  
    🔎 *Note: This is a rough estimate based on pile size and typical material density.*
    """)

else:
    st.warning("👈 Please upload an image from the sidebar to start prediction!")
