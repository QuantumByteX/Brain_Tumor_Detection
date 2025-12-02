import streamlit as st
from PIL import Image
import numpy as np
import cv2

@st.cache_resource
def Model_load(name):
    from tensorflow.keras.models import load_model
    return load_model(name)

def preprocess_image(image):
    img_arr = np.array(image)
    img = cv2.resize(img_arr, (224,224))

    if img.shape != (224,224,3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = img / 255.0
    img = np.expand_dims(img, 0)
    return img

model = Model_load("brain_tumor_Detection_01.keras")

class_label = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.set_page_config(page_title="Tumor Detector", page_icon="🧠")

st.title("Brain Tumor Detection System !!")
st.write("( Upload an MRI scan of brain and the model will predict whether tumor is present or not. )")
st.text('')
st.text('')
uploaded_file = st.file_uploader("Upload an MRI Image (🧠):", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)

            class_ind = np.argmax(prediction)
            class_label_fetch = class_label[class_ind]
            result = f"Tumor Detected!! Type : {class_label_fetch}" if class_ind in [0,1,3] else "No Tumor"
            st.success(f"Prediction: {result}")