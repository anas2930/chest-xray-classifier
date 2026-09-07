import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

import base64

def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("C:/Users/FASTECH LAPTOP/Downloads/download.jpeg")

OPTIMAL_THRESHOLD = 0.9990534
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

st.set_page_config(page_title="Chest X-Ray Classifier", 
                   page_icon="🫁", 
                   layout="wide")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/vgg16_finetuned.h5')

model = load_model()

def preprocess_xray(image_bytes, target_size=(224, 224), train_mean=0.482, train_std=0.236):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_h, pad_w = target_size[0] - new_h, target_size[1] - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    image = image.astype(np.float32) / 255.0
    image = (image - train_mean) / train_std
    image = np.stack([image] * 3, axis=-1)
    return np.expand_dims(image, axis=0)

st.markdown("<h1 style='text-align: center; font-size: 80px; color: yellow;'>Chest X-Ray Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 30px; color: white;'>CNN VGG-16 BASED CLASSIFIER</p>", unsafe_allow_html=True)

#st.title("CHEST X-RAY CLASSIFIER 🫁")
#st.caption("<FINE TUNED VGG-16 BASED CLASSIFIER>",)

#uploaded_file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

#uploaded_file=st.file_uploader(st.markdown("<p font-size: 10px; color: gray;'>>UPLOAD FILE<</p>", unsafe_allow_html=True), type=["jpg", "jpeg", "png"])

st.divider()

#st.markdown("<p style='text-align: center; font-size: 16px; color: gray;'>UPLOAD FILE</p>", unsafe_allow_html=True)

#uploaded_file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png"])

uploaded_file = st.file_uploader("UPLOAD CHEST X-RAY IMAGE", type=["jpg", "jpeg", "png"])

st.markdown("""
    <style>
    h1, h2, h3, p, .stMarkdown {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


st.sidebar.title("DESCRIPTION")

st.sidebar.write("MODEL: VGG16")

st.sidebar.write("STATUS: FINE TUNED")

st.sidebar.write("INPUT: 224 × 224")

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Uploaded X-ray Image", use_container_width=True,width=10)

    if st.button("Predict", type="primary",width=200):
        with st.spinner("Predicting..."):
            processed = preprocess_xray(image_bytes)
            prob = float(model.predict(processed, verbose=0)[0][0])
            prediction = CLASS_NAMES[1] if prob > OPTIMAL_THRESHOLD else CLASS_NAMES[0]

        if prediction == "NORMAL":
            st.success(f"Result: {prediction}")
        else:
            st.error(f"Result: {prediction}")

        
