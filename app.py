
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Mango Disease Detection",
    page_icon="🥭",
    layout="centered"
)

st.title("🥭 Mango Plant Disease Detection")
st.write("Upload a mango leaf image for CNN-based disease detection.")

CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

ADVICE = {
    "Anthracnose": "Remove infected leaves and ask an agricultural expert about suitable fungicide treatment.",
    "Bacterial Canker": "Prune infected parts using sanitized tools and seek expert advice.",
    "Cutting Weevil": "Inspect damaged shoots and seek appropriate pest-management advice.",
    "Die Back": "Prune dead branches with sanitized tools and investigate the cause.",
    "Gall Midge": "Inspect affected leaves and shoots and seek local pest-management advice.",
    "Healthy": "No disease was identified by the model. Continue regular plant care.",
    "Powdery Mildew": "Improve airflow and seek advice about suitable disease control.",
    "Sooty Mould": "Check for sap-feeding insects and manage the underlying infestation."
}

@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="disha-tadas/mango-disease-model",
        filename="mango_disease_model.keras"
    )
    return tf.keras.models.load_model(path)

uploaded_file = st.file_uploader(
    "Upload a mango leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    if st.button("Detect Disease", type="primary"):
        try:
            with st.spinner("Analyzing image..."):
                model = load_model()

                # Check the model's actual expected input size.
                input_shape = model.input_shape
                height = input_shape[1] or 224
                width = input_shape[2] or 224

                resized = image.resize((width, height))
                array = np.asarray(resized, dtype=np.float32)

                # Use raw RGB pixels, matching the earlier app pipeline.
                # If training used different preprocessing, this must change.
                array = np.expand_dims(array, axis=0)

                output = model.predict(array, verbose=0)[0]

                if len(output) != len(CLASS_NAMES):
                    st.error(
                        f"Expected 8 class outputs, but model returned {len(output)}."
                    )
                else:
                    index = int(np.argmax(output))
                    disease = CLASS_NAMES[index]
                    score = float(output[index])

                    st.subheader("Detection Result")
                    st.success(f"Predicted class: {disease}")
                    st.metric("Model score", f"{score * 100:.2f}%")

                    st.write("### Suggested next step")
                    st.write(ADVICE[disease])

                    st.caption(
                        "The model score is not a validated measure of "
                        "disease severity or the percentage of leaf affected."
                    )

                    with st.expander("View scores for all classes"):
                        for name, value in zip(CLASS_NAMES, output):
                            st.write(f"**{name}:** {float(value) * 100:.2f}%")

        except Exception as error:
            st.error(f"Prediction failed: {error}")
