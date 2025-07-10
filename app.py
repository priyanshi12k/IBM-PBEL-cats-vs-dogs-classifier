import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Load trained model
model = load_model("best_model.keras")

# Page config
st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐶", layout="centered")
st.title("🐱 Cats vs Dogs Classifier 🐶")
st.markdown("Upload or paste an image to classify it as a **Cat** or **Dog** using a CNN model trained with Keras.")

# Preprocessing function
def preprocess(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# --- Upload Options ---
st.markdown("### 📤 Choose an Input Method")
tab1, tab2 = st.tabs(["📁 Upload Image", "🌐 Paste Image URL"])

img = None

# 📁 Upload from local (supports drag, browse, paste)
with tab1:
    uploaded_file = st.file_uploader(
        "Upload an image (📂 drag & drop, 🖱️ browse, or 📋 paste)",
        type=["jpg", "jpeg", "png"]
    )
    st.caption("💡 Tip: Click above and press **Ctrl+V** to paste an image directly!")
    if uploaded_file:
        img = Image.open(uploaded_file)

# 🌐 URL Input
with tab2:
    img_url = st.text_input("Paste an image URL (ends with .jpg / .png):")
    if img_url:
        try:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
        except:
            st.error("⚠️ Could not load image from the provided URL.")

# --- Prediction ---
if img:
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🧠 Predicting..."):
        processed = preprocess(img)
        prediction = model.predict(processed)[0][0]
        confidence = float(prediction)

        label = "Dog 🐶" if confidence > 0.5 else "Cat 🐱"
        prob = confidence if confidence > 0.5 else 1 - confidence

        st.success(f"### ✅ Prediction: **{label}**")
        st.progress(prob)
        st.markdown(f"**Confidence:** `{prob:.2%}`")

else:
    st.info("👆 Upload an image or paste a link above to get started.")
