import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import requests


# =========================
# MODEL
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mango_disease_model.keras")


model = load_model()


# =========================
# CLASS NAMES
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
        "treatment": "Remove infected leaves and affected plant parts and follow recommended fungicide treatment.",
        "prevention": "Maintain good air circulation, avoid excess moisture and remove infected plant material.",
        "fertilizer": "Maintain balanced plant nutrition and avoid excessive nitrogen."
    },

    "Bacterial Canker": {
        "severity": "High",
        "cause": "Bacterial infection causing lesions and cankers.",
        "treatment": "Remove severely infected parts and follow recommended disease-management practices.",
        "prevention": "Use healthy planting material and maintain proper sanitation.",
        "fertilizer": "Use balanced NPK nutrition according to soil requirements."
    },

    "Cutting Weevil": {
        "severity": "Medium",
        "cause": "Damage caused by cutting weevil infestation.",
        "treatment": "Remove badly damaged plant parts and follow recommended pest-control practices.",
        "prevention": "Regularly inspect new shoots and remove affected material.",
        "fertilizer": "Maintain balanced NPK nutrition for healthy plant growth."
    },

    "Die Back": {
        "severity": "High",
        "cause": "Disease causing drying and death of shoots and branches.",
        "treatment": "Prune affected branches and follow recommended disease-management practices.",
        "prevention": "Remove dead or infected branches and maintain orchard hygiene.",
        "fertilizer": "Use balanced fertilizer according to soil-test recommendations."
    },

    "Gall Midge": {
        "severity": "Medium",
        "cause": "Insect infestation causing damage to young leaves and shoots.",
        "treatment": "Remove severely affected plant parts and follow recommended insect-management practices.",
        "prevention": "Regularly monitor young leaves and shoots.",
        "fertilizer": "Maintain balanced NPK nutrition for healthy plant growth."
    },

    "Healthy": {
        "severity": "None",
        "cause": "No major disease detected.",
        "treatment": "No disease treatment is required.",
        "prevention": "Continue regular watering, nutrition and routine monitoring.",
        "fertilizer": "Continue balanced fertilizer application according to soil requirements."
    },

    "Powdery Mildew": {
        "severity": "High",
        "cause": "Fungal infection producing white powdery growth.",
        "treatment": "Remove severely affected parts and follow recommended disease-management practices.",
        "prevention": "Maintain good air circulation and avoid excessive humidity.",
        "fertilizer": "Avoid excessive nitrogen and maintain balanced plant nutrition."
    },

    "Sooty Mould": {
        "severity": "Medium",
        "cause": "Dark fungal growth commonly associated with sucking insects.",
        "treatment": "Control associated insect infestation and clean affected plant surfaces.",
        "prevention": "Monitor for sucking insects and maintain good plant hygiene.",
        "fertilizer": "Maintain balanced nutrition and manage associated insect infestation."
    }
}


# =========================
# SPOT HIGHLIGHTING
# =========================

def highlight_spots(image):

    img = np.array(image.convert("RGB"))

    # OpenCV format
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Detect darker abnormal spots
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    difference = blurred.astype(np.int16) - gray.astype(np.int16)

    # Leaf-like region
    saturation = hsv[:, :, 1]

    leaf_mask = (
        (saturation > 25) &
        (gray < 240)
    ).astype(np.uint8) * 255

    # Candidate disease spots
    spots = (
        (difference > 12) &
        (gray < 180) &
        (leaf_mask > 0)
    ).astype(np.uint8) * 255

    # Remove tiny noise
    kernel = np.ones((3, 3), np.uint8)

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

    # Find individual spots
    contours, _ = cv2.findContours(
        spots,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    number_of_spots = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        # Keep only visible disease-like spots
        if 15 <= area <= 800:

            number_of_spots += 1

            x, y, w, h = cv2.boundingRect(contour)

            # Red outline around spot
            cv2.drawContours(
                result,
                [contour],
                -1,
                (255, 0, 0),
                2
            )

            # Small red center
            cv2.circle(
                result,
                (x + w // 2, y + h // 2),
                3,
                (255, 0, 0),
                -1
            )

    return result, number_of_spots


# =========================
# WEATHER
# =========================

def get_weather(city):

    try:

        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        data = response.json()

        if "results" not in data:
            return None

        location = data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation"
        }

        response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        current = response.json()["current"]

        return current

    except Exception:
        return None


# =========================
# PAGE DESIGN
# =========================

st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon="🥭",
    layout="centered"
)

st.title("🥭 Mango Leaf Disease Detection")

st.write(
    "Upload a mango leaf image to automatically detect the disease."
)


# =========================
# UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "📷 Choose a mango leaf image",
    type=["jpg", "jpeg", "png"]
)


# =========================
# DETECTION
# =========================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Mango Leaf",
        width=500
    )

    # Resize for CNN
    resized = image.resize((224, 224))

    img_array = np.array(resized)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = int(
        np.argmax(prediction[0])
    )

    predicted_class = class_names[predicted_index]

    confidence = float(
        np.max(prediction[0]) * 100
    )

    info = disease_info[predicted_class]


    # =========================
    # RESULT
    # =========================

    st.success(
        f"🌿 Predicted Disease: {predicted_class}"
    )

    st.info(
        f"🎯 Confidence: {confidence:.2f}%"
    )


    # =========================
    # DISEASE INFORMATION
    # =========================

    st.header("🩺 Disease Information")

    st.write(
        f"**Severity:** {info['severity']}"
    )

    st.write(
        f"**Cause:** {info['cause']}"
    )

    st.write(
        f"**Treatment:** {info['treatment']}"
    )

    st.write(
        f"**Prevention:** {info['prevention']}"
    )


    # =========================
    # FERTILIZER
    # =========================

    st.header("🌱 Fertilizer Recommendation")

    st.write(
        info["fertilizer"]
    )

    st.caption(
        "General recommendation. Actual fertilizer application should follow soil-test and local agricultural guidance."
    )


    # =========================
    # DISEASED SPOTS
    # =========================

    st.header("🔍 Diseased Area Highlighting")

    highlighted_image, spot_count = highlight_spots(image)

    st.image(
        highlighted_image,
        caption="Detected Abnormal/Diseased Spots",
        width=500
    )

    if spot_count > 0:

        st.success(
            f"Approximately {spot_count} abnormal spot(s) detected."
        )

    else:

        st.warning(
            "No clear individual spots were detected in this image."
        )


    # =========================
    # WEATHER
    # =========================

    st.header("🌦️ Weather-Based Recommendation")

    city = st.text_input(
        "Enter your city",
        value="Bengaluru"
    )

    if st.button("Check Weather"):

        weather = get_weather(city)

        if weather is not None:

            temperature = weather["temperature_2m"]
            humidity = weather["relative_humidity_2m"]
            rain = weather["precipitation"]

            st.write(
                f"🌡️ **Temperature:** {temperature} °C"
            )

            st.write(
                f"💧 **Humidity:** {humidity}%"
            )

            st.write(
                f"🌧️ **Precipitation:** {rain} mm"
            )

            if humidity >= 80:

                st.warning(
                    "High humidity. Maintain good air circulation and monitor fungal diseases."
                )

            elif rain > 0:

                st.warning(
                    "Rainfall is present. Avoid unnecessary leaf wetting."
                )

            else:

                st.success(
                    "Weather conditions are suitable for routine plant monitoring."
                )

        else:

            st.error(
                "Unable to get weather information."
            )


# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Mango Leaf Disease Detection using CNN | Major Project"
)