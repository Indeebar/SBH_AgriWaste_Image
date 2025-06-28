import streamlit as st
import tensorflow as tf
import numpy as np
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
st.markdown("Upload an image and let our ResNet model classify the type of agricultural waste!")

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

# ----------------- MODEL PATH -----------------
classifier_model_url = 'https://drive.google.com/uc?id=1IoofyBzkSRMpo0P7DEzOciJVyvlpiVQZ'
classifier_model_path = 'agri_waste_classifier_resnet.h5'

# ----------------- DOWNLOAD AND LOAD -----------------
download_model(classifier_model_url, classifier_model_path)
classifier_model = load_model(classifier_model_path)

# ----------------- IF IMAGE IS UPLOADED -----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner('Predicting waste type...'):
        prediction = classifier_model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]

    # Show result
    st.success(f"🎯 **Predicted Waste Type:** {predicted_class}")

else:
    st.warning("👈 Please upload an image from the sidebar to start prediction!")

