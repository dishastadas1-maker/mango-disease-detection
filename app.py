
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Mango Disease Detection", page_icon="🥭")

st.title("🥭 Mango Plant Disease Detection")
st.write("Upload a mango leaf image to detect its disease using CNN.")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="disha-tadas/mango-disease-model",
        filename="mango_disease_model.keras"
    )
    return tf.keras.models.load_model(model_path)

class_names = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

advice = {
    "Anthracnose": "Remove infected plant parts and seek suitable fungicide advice.",
    "Bacterial Canker": "Prune infected parts with sanitized tools and consult an agricultural expert.",
    "Cutting Weevil": "Inspect affected shoots and seek local pest-management advice.",
    "Die Back": "Prune dead branches and investigate the underlying cause.",
    "Gall Midge": "Remove severely affected plant parts and seek pest-management advice.",
    "Healthy": "No disease detected by the model. Continue regular plant care.",
    "Powdery Mildew": "Improve airflow and seek suitable fungicide advice.",
    "Sooty Mould": "Control sap-feeding insects and gently clean affected leaves."
}

uploaded_file = st.file_uploader(
    "Choose a mango leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    if st.button("Detect Disease"):
        try:
            with st.spinner("Analyzing leaf..."):
                model = load_model()

                # Assumes the model was trained with 224 x 224 RGB images.
                img = image.resize((224, 224))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array, verbose=0)[0]
                index = int(np.argmax(prediction))

                if len(prediction) != len(class_names):
                    st.error("Model output does not match the 8 disease classes.")
                else:
                    disease = class_names[index]
                    confidence = float(prediction[index]) * 100

                    st.subheader("Detection Result")
                    st.success(f"Disease: {disease}")
                    st.write(f"Model confidence: {confidence:.2f}%")
                    st.write("**Suggested next step:**")
                    st.write(advice[disease])

        except Exception as e:
            st.error(f"Error: {e}")
