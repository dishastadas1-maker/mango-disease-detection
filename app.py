import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import requests
from pathlib import Path

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon="🥭",
    layout="centered"
)

# =========================
# MODEL LOADING
# =========================

@st.cache_resource
def load_model():
    local_path = Path("mango_disease_model.keras")

    if local_path.exists():
        model_path = str(local_path)
    else:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="disha-tadas/mango-disease-model",
            filename="mango_disease_model.keras"
        )

    return tf.keras.models.load_model(model_path)


try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the mango disease model.")
    st.code(str(error))
    st.stop()

# =========================
# DISEASE CLASSES
# =========================

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

# =========================
# DISEASE INFORMATION
# =========================

disease_info = {
    "Anthracnose": {
        "severity": "High",
        "cause": "Fungal infection caused by Colletotrichum species.",
        "treatment": "Remove infected plant material and confirm the disease before choosing a fungicide.",
        "prevention": "Maintain airflow, avoid excess leaf moisture and remove fallen infected leaves.",
        "fertilizer": "Maintain balanced plant nutrition based on soil requirements."
    },
    "Bacterial Canker": {
        "severity": "High",
        "cause": "Bacterial infection that may cause lesions and cankers.",
        "treatment": "Remove severely affected parts where appropriate and seek agricultural advice.",
        "prevention": "Use healthy planting material and sanitize pruning tools.",
        "fertilizer": "Use balanced nutrients based on soil testing."
    },
    "Cutting Weevil": {
        "severity": "Medium",
        "cause": "Plant damage associated with cutting weevil infestation.",
        "treatment": "Confirm the pest and seek advice about suitable registered pest-control options.",
        "prevention": "Inspect new shoots regularly and remove badly damaged material.",
        "fertilizer": "Maintain balanced plant nutrition."
    },
    "Die Back": {
        "severity": "High",
        "cause": "Drying and death of shoots or branches.",
        "treatment": "Prune dead branches using sanitized tools and confirm the cause of the damage.",
        "prevention": "Maintain orchard hygiene and avoid plant stress.",
        "fertilizer": "Use fertilizer according to soil-test recommendations."
    },
    "Gall Midge": {
        "severity": "Medium",
        "cause": "Insect infestation that can damage young leaves and shoots.",
        "treatment": "Confirm the insect before selecting a registered treatment.",
        "prevention": "Monitor young leaves and shoots regularly.",
        "fertilizer": "Maintain balanced nutrients for healthy growth."
    },
    "Healthy": {
        "severity": "None detected",
        "cause": "The model classified the leaf as healthy.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Continue routine monitoring, proper watering and plant hygiene.",
        "fertilizer": "Apply balanced fertilizer according to soil requirements."
    },
    "Powdery Mildew": {
        "severity": "High",
        "cause": "Fungal infection that produces white, powder-like growth.",
        "treatment": "Confirm the disease and consult an agricultural expert about a suitable registered fungicide.",
        "prevention": "Maintain airflow and monitor plants regularly.",
        "fertilizer": "Avoid excessive nitrogen and maintain balanced nutrition."
    },
    "Sooty Mould": {
        "severity": "Medium",
        "cause": "Dark fungal growth often associated with honeydew-producing insects.",
        "treatment": "Identify and manage the insects responsible for the honeydew.",
        "prevention": "Monitor for scale insects, aphids and other sap-feeding pests.",
        "fertilizer": "Maintain balanced nutrition according to soil requirements."
    }
}

# =========================
# MEDICINE / PESTICIDE
# RECOMMENDATIONS
# =========================

medicine_info = {
    "Anthracnose": [
        "Copper-based fungicides may be an option if registered for mango anthracnose in your area.",
        "Ask an agricultural officer to confirm the diagnosis and treatment."
    ],
    "Bacterial Canker": [
        "Confirm the bacterial disease before treatment.",
        "Ask an agricultural officer about locally recommended management options."
    ],
    "Cutting Weevil": [
        "Confirm the pest before choosing an insecticide.",
        "Use only a product registered for mango and the identified pest."
    ],
    "Die Back": [
        "Prune dead branches using sanitized tools.",
        "Use a fungicide only if the cause is confirmed and the product is registered for that use."
    ],
    "Gall Midge": [
        "Confirm the infestation before selecting an insect-control product.",
        "Ask an agricultural officer for a locally registered treatment."
    ],
    "Healthy": [
        "No pesticide is recommended based on a healthy classification."
    ],
    "Powdery Mildew": [
        "Wettable sulphur may be an option if registered for mango powdery mildew in your area.",
        "Follow the product label and seek agricultural advice."
    ],
    "Sooty Mould": [
        "Identify and manage the insects producing honeydew.",
        "Ask an agricultural officer to recommend a suitable registered pest-control product."
    ]
}

# =========================
# CNN PREDICTION
# =========================

def predict_disease(image):
    resized = image.resize((224, 224))

    image_array = np.array(
        resized,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

    index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][index]) * 100

    return class_names[index], confidence

# =========================
# OPENCV SPOT HIGHLIGHTING
# =========================

def highlight_spots(image):
    rgb = np.array(image.convert("RGB"))

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    hsv = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (21, 21),
        0
    )

    difference = (
        blurred.astype(np.int16)
        - gray.astype(np.int16)
    )

    saturation = hsv[:, :, 1]

    leaf_mask = (
        (saturation > 25) &
        (gray < 240)
    ).astype(np.uint8) * 255

    spots = (
        (difference > 12) &
        (gray < 180) &
        (leaf_mask > 0)
    ).astype(np.uint8) * 255

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    spots = cv2.morphologyEx(
        spots,
        cv2.MORPH_OPEN,
        kernel
    )

    spots = cv2.morphologyEx(
        spots,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        spots,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = rgb.copy()
    spot_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if 15 <= area <= 800:
            cv2.drawContours(
                result,
                [contour],
                -1,
                (255, 0, 0),
                2
            )

            spot_count += 1

    return Image.fromarray(result), spot_count

# =========================
# WEATHER FUNCTION
# =========================

def get_weather(city):
    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            },
            timeout=10
        )

        geo_response.raise_for_status()
        locations = geo_response.json().get("results", [])

        if not locations:
            return None

        location = locations[0]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "precipitation"
                ),
                "timezone": "auto"
            },
            timeout=10
        )

        weather_response.raise_for_status()

        current = weather_response.json().get("current")

        if not current:
            return None

        return {
            "place": location.get("name", city),
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "rain": current.get("precipitation")
        }

    except (requests.RequestException, ValueError, KeyError):
        return None

# =========================
# APP INTERFACE
# =========================

st.title("🥭 Mango Leaf Disease Detection")

st.write(
    "Upload a mango leaf image to detect its disease, "
    "view treatment and fertilizer suggestions, highlight "
    "possible affected spots, and check local weather."
)

st.info(
    "AI predictions are estimates. Confirm the disease "
    "before applying any pesticide or treatment."
)

# =========================
# IMAGE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "📷 Choose a mango leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Mango Leaf",
            use_container_width=True
        )

        if st.button("🔍 Detect Disease", type="primary"):
            with st.spinner("Analysing the mango leaf..."):
                predicted_class, confidence = predict_disease(image)

                highlighted_image, spot_count = highlight_spots(image)

            st.session_state["result"] = {
                "disease": predicted_class,
                "confidence": confidence,
                "image": image,
                "highlighted": highlighted_image,
                "spot_count": spot_count
            }

    except Exception as error:
        st.error("Unable to process this image.")
        st.code(str(error))

# =========================
# DISPLAY RESULTS
# =========================

if "result" in st.session_state:
    result = st.session_state["result"]

    predicted_class = result["disease"]
    confidence = result["confidence"]
    info = disease_info[predicted_class]

    st.divider()
    st.header("🌿 Detection Result")

    st.success(f"Predicted Disease: {predicted_class}")

    st.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )

    if confidence < 60:
        st.warning(
            "The confidence is relatively low. Try a clearer image "
            "with good lighting and a visible leaf."
        )

    # Disease details
    st.header("🩺 Disease Information")

    st.write(f"**Severity:** {info['severity']}")
    st.write(f"**Cause:** {info['cause']}")
    st.write(f"**Treatment:** {info['treatment']}")
    st.write(f"**Prevention:** {info['prevention']}")

    # Medicine recommendations
    st.header("💊 Medicine / Pesticide Recommendations")

    for medicine in medicine_info[predicted_class]:
        st.write(f"• {medicine}")

    st.caption(
        "General guidance only. Confirm the diagnosis and use only "
        "products registered for mango and the identified disease or pest. "
        "Follow the product label, protective-equipment instructions "
        "and pre-harvest interval."
    )

    # Fertilizer
    st.header("🌱 Fertilizer Recommendation")

    st.write(info["fertilizer"])

    st.caption(
        "Actual fertilizer application should follow soil testing "
        "and local agricultural guidance."
    )

    # Highlighting
    st.header("🔴 Possible Affected Spots")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            result["image"],
            caption="Original Leaf",
            use_container_width=True
        )

    with col2:
        st.image(
            result["highlighted"],
            caption="Possible Spots Highlighted",
            use_container_width=True
        )

    st.write(
        f"**Possible regions highlighted:** {result['spot_count']}"
    )

    st.caption(
        "OpenCV highlights image regions based on image processing. "
        "These regions are not confirmed disease spots or a validated "
        "measurement of disease severity."
    )

    if predicted_class == "Healthy":
        st.success(
            "The model classified this leaf as healthy. "
            "Continue routine monitoring."
        )
    else:
        st.warning(
            "Inspect the plant and confirm the diagnosis before treatment."
        )

# =========================
# WEATHER SECTION
# =========================

st.divider()
st.header("🌦️ Weather Information")

city = st.text_input(
    "Enter your city",
    value="Bengaluru"
)

if st.button("Check Weather"):
    if not city.strip():
        st.warning("Please enter a city name.")
    else:
        with st.spinner("Fetching weather information..."):
            weather = get_weather(city.strip())

        if weather:
            st.subheader(f"Weather for {weather['place']}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Temperature",
                    f"{weather['temperature']} °C"
                    if weather["temperature"] is not None
                    else "Unavailable"
                )

            with col2:
                st.metric(
                    "Humidity",
                    f"{weather['humidity']}%"
                    if weather["humidity"] is not None
                    else "Unavailable"
                )

            with col3:
                st.metric(
                    "Precipitation",
                    f"{weather['rain']} mm"
                    if weather["rain"] is not None
                    else "Unavailable"
                )

            humidity = weather["humidity"]
            rain = weather["rain"]

            if humidity is not None and humidity >= 80:
                st.warning(
                    "Humidity is high. Monitor for fungal diseases "
                    "and maintain good airflow."
                )
            elif rain is not None and rain > 0:
                st.info(
                    "Precipitation is present. Avoid unnecessary "
                    "leaf wetting and monitor the plant."
                )
            else:
                st.success(
                    "Continue routine plant monitoring and care."
                )

            st.caption(
                "Weather data provided by Open-Meteo."
            )

        else:
            st.error(
                "Unable to retrieve weather. Check the city name "
                "and internet connection."
            )

# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "Mango Leaf Disease Detection using CNN | Major Project"
)
