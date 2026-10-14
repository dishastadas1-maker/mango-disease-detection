
import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from huggingface_hub import hf_hub_download
import cv2

st.set_page_config(
    page_title="Mango Disease Detection",
    page_icon="🥭",
    layout="wide"
)

st.title("🥭 Mango Plant Disease Detection")
st.write("CNN-based mango leaf disease detection and plant-care support.")

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
    "Anthracnose": (
        "Remove badly infected fallen leaves. Keep foliage dry and ask an "
        "agricultural expert about locally approved fungicide options."
    ),
    "Bacterial Canker": (
        "Prune affected parts with sanitized tools when appropriate. Avoid "
        "moving contaminated plant material and consult an agricultural expert."
    ),
    "Cutting Weevil": (
        "Inspect shoots for cutting or damage. Remove heavily damaged material "
        "where appropriate and seek local pest-management advice."
    ),
    "Die Back": (
        "Prune dead branches with sanitized tools and investigate possible "
        "pests, disease, water stress, or root problems."
    ),
    "Gall Midge": (
        "Inspect affected leaves and shoots. Remove badly affected material "
        "where appropriate and seek local pest-management advice."
    ),
    "Healthy": (
        "The model predicts a healthy leaf. Continue regular watering, "
        "monitoring, and balanced plant care."
    ),
    "Powdery Mildew": (
        "Improve airflow and avoid excessive leaf wetness. Ask an agricultural "
        "expert about locally approved disease-control options."
    ),
    "Sooty Mould": (
        "Inspect for sap-feeding insects such as scale insects or aphids. "
        "Address the underlying pest problem using locally appropriate advice."
    )
}

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="disha-tadas/mango-disease-model",
        filename="mango_disease_model.keras"
    )
    return tf.keras.models.load_model(model_path)

@st.cache_data(ttl=1800)
def get_weather(place):
    # Geocode the place name using Open-Meteo.
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_response = requests.get(
        geo_url,
        params={"name": place, "count": 1, "language": "en", "format": "json"},
        timeout=15
    )
    geo_response.raise_for_status()
    geo_results = geo_response.json().get("results", [])

    if not geo_results:
        return None

    location = geo_results[0]
    lat = location["latitude"]
    lon = location["longitude"]

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,"
                       "precipitation,weather_code,wind_speed_10m",
            "timezone": "auto"
        },
        timeout=15
    )
    weather_response.raise_for_status()

    return location, weather_response.json().get("current", {})

def weather_description(code):
    if code == 0:
        return "Clear sky"
    if code in (1, 2, 3):
        return "Mainly clear to cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55, 56, 57):
        return "Drizzle"
    if code in (61, 63, 65, 66, 67):
        return "Rain"
    if code in (71, 73, 75, 77):
        return "Snow"
    if code in (80, 81, 82):
        return "Rain showers"
    if code in (95, 96, 99):
        return "Thunderstorm"
    return "Conditions unavailable"

def estimate_leaf_areas(pil_image):
    """
    Rough visual estimate only. This is not a trained lesion-segmentation model.
    It can mistake shadows, veins, background, or natural colour changes for disease.
    """
    rgb = np.array(pil_image.convert("RGB"))
    h, w = rgb.shape[:2]

    # Resize for faster processing while preserving the original image for display.
    scale = min(1.0, 700 / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(
            rgb, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Broad green/yellow/brown colour range to estimate likely leaf pixels.
    leaf_mask = (
        (hsv[:, :, 0] >= 15) &
        (hsv[:, :, 0] <= 100) &
        (hsv[:, :, 1] >= 25) &
        (hsv[:, :, 2] >= 25)
    )

    leaf_count = int(np.count_nonzero(leaf_mask))
    if leaf_count < 100:
        return None, None

    # Mark dark brown, reddish, and pale discoloured pixels within likely leaf.
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    brown_or_dark = (
        (r > g * 0.85) &
        (r > b * 0.8) &
        ((r - g) > 8) &
        (value < 190)
    )
    pale_discolouration = (saturation < 65) & (value > 100)
    candidate = leaf_mask & (brown_or_dark | pale_discolouration)

    # Reduce isolated noise.
    candidate_u8 = (candidate.astype(np.uint8) * 255)
    candidate_u8 = cv2.medianBlur(candidate_u8, 5)
    candidate = candidate_u8 > 0

    affected_count = int(np.count_nonzero(candidate & leaf_mask))
    affected_percent = min(100.0, affected_count / leaf_count * 100.0)

    # Make a visual overlay of candidate discoloured areas.
    overlay = rgb.copy()
    overlay[candidate & leaf_mask] = [255, 0, 0]
    highlighted = Image.fromarray(overlay)

    return affected_percent, highlighted

# Weather section
st.subheader("🌦️ Current Weather")
place = st.text_input("Enter your city or town", value="Bengaluru")

if st.button("Get Current Weather"):
    try:
        with st.spinner("Getting current weather..."):
            result = get_weather(place.strip())

        if result is None:
            st.warning("Location not found. Try a different city or town name.")
        else:
            location, current = result
            st.write(
                f"**Location:** {location['name']}, "
                f"{location.get('country', '')}"
            )
            if current:
                c1, c2, c3 = st.columns(3)
                c1.metric("Temperature", f"{current.get('temperature_2m', '—')} °C")
                c2.metric("Humidity", f"{current.get('relative_humidity_2m', '—')}%")
                c3.metric("Wind speed", f"{current.get('wind_speed_10m', '—')} km/h")
                st.write(
                    "Conditions:",
                    weather_description(current.get("weather_code", -1))
                )
                st.caption(
                    "Weather data: Open-Meteo. Weather is general local context, "
                    "not a diagnosis of plant disease."
                )
            else:
                st.warning("Weather data is currently unavailable.")
    except Exception as error:
        st.warning(f"Could not retrieve weather: {error}")

st.divider()
st.subheader("🔬 Leaf Disease Analysis")

uploaded_file = st.file_uploader(
    "Upload a clear mango leaf photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded mango leaf", use_container_width=True)

    if st.button("Detect Disease", type="primary"):
        try:
            with st.spinner("Analyzing leaf with CNN..."):
                model = load_model()

                # Use the model's expected input dimensions.
                shape = model.input_shape
                height = shape[1] or 224
                width = shape[2] or 224

                resized = image.resize((width, height))
                array = np.asarray(resized, dtype=np.float32)

                # Preserve the raw-RGB preprocessing that worked in your latest test.
                array = np.expand_dims(array, axis=0)
                output = np.asarray(
                    model.predict(array, verbose=0)[0]
                ).flatten()

                if len(output) != len(CLASS_NAMES):
                    st.error(
                        f"Expected 8 class outputs, received {len(output)}."
                    )
                    st.stop()

                index = int(np.argmax(output))
                disease = CLASS_NAMES[index]
                score = float(output[index])

                st.subheader("Detection Result")
                st.success(f"Predicted disease: {disease}")
                st.metric("Model score", f"{score * 100:.2f}%")
                st.write("### Prevention and care")
                st.write(ADVICE[disease])

                affected_percent, highlighted = estimate_leaf_areas(image)

                st.write("### Severity estimate")
                if disease == "Healthy":
                    st.info(
                        "The model predicts a healthy leaf. The visual colour "
                        "estimate is not a disease-severity measurement."
                    )
                elif affected_percent is None:
                    st.warning(
                        "Could not estimate discoloured leaf area from this image. "
                        "Try a clear photo with the leaf filling most of the frame."
                    )
                else:
                    st.metric(
                        "Estimated discoloured area of likely leaf pixels",
                        f"{affected_percent:.1f}%"
                    )
                    if affected_percent < 5:
                        st.write("Visual estimate: low discolouration")
                    elif affected_percent < 20:
                        st.write("Visual estimate: moderate discolouration")
                    else:
                        st.write("Visual estimate: high discolouration")

                    st.caption(
                        "This is a rough colour-based estimate, not a validated "
                        "measurement of infected tissue or disease severity."
                    )

                if highlighted is not None:
                    st.write("### Highlighted leaf areas")
                    st.image(
                        highlighted,
                        caption=(
                            "Red marks show candidate discoloured areas, "
                            "not confirmed disease lesions."
                        ),
                        use_container_width=True
                    )

                with st.expander("Scores for all classes"):
                    for name, value in zip(CLASS_NAMES, output):
                        st.write(f"**{name}:** {float(value) * 100:.2f}%")

        except Exception as error:
            st.error(f"Prediction failed: {error}")
