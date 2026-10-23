
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import requests
from huggingface_hub import hf_hub_download

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon="🥭",
    layout="centered"
)

# ==================================================
# MODEL
# ==================================================

MODEL_REPO = "disha-tadas/mango-disease-model"
MODEL_FILENAME = "mango_disease_model.keras"


@st.cache_resource
def load_model():
    # Use a local model if it exists; otherwise download it
    # from the Hugging Face repository.
    if os.path.exists(MODEL_FILENAME):
        model_path = MODEL_FILENAME
    else:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME
        )

    return tf.keras.models.load_model(model_path)


# ==================================================
# CLASS NAMES — KEEP IN THE TRAINED MODEL'S ORDER
# ==================================================

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

# ==================================================
# DISEASE INFORMATION
# ==================================================

disease_info = {
    "Anthracnose": {
        "severity": "High",
        "affected": "Leaves, flowers, twigs and fruits",
        "cause": "Fungal infection caused by Colletotrichum species.",
        "treatment": (
            "Remove infected plant material. A suitable fungicide may be "
            "needed after confirming the diagnosis."
        ),
        "prevention": (
            "Maintain good air circulation, avoid prolonged leaf wetness "
            "and remove infected plant material."
        )
    },

    "Bacterial Canker": {
        "severity": "High",
        "affected": "Leaves, twigs, branches and sometimes fruits",
        "cause": "Bacterial infection that can cause lesions and cankers.",
        "treatment": (
            "Prune severely affected parts using clean tools and seek "
            "local agricultural advice for appropriate disease management."
        ),
        "prevention": (
            "Use healthy planting material, maintain sanitation and avoid "
            "injuring plant tissues."
        )
    },

    "Cutting Weevil": {
        "severity": "Medium",
        "affected": "Young shoots, twigs and branches",
        "cause": "Damage associated with cutting weevil infestation.",
        "treatment": (
            "Inspect affected shoots, remove badly damaged material and "
            "confirm the pest before selecting a control product."
        ),
        "prevention": (
            "Inspect new shoots regularly and remove affected plant material."
        )
    },

    "Die Back": {
        "severity": "High",
        "affected": "Twigs, branches and shoot tips",
        "cause": "A condition that causes shoots and branches to dry and die; "
                 "fungal infection may be involved.",
        "treatment": (
            "Prune dead or affected branches back to healthy tissue using "
            "clean tools. Confirm the cause before applying a treatment."
        ),
        "prevention": (
            "Maintain orchard hygiene, avoid plant injuries and remove dead "
            "branches."
        )
    },

    "Gall Midge": {
        "severity": "Medium",
        "affected": "Young leaves, shoots, buds or flowers, depending on the pest",
        "cause": "Insect infestation that may damage young plant growth.",
        "treatment": (
            "Inspect the damaged plant part and confirm the pest species "
            "before choosing an insect-control product."
        ),
        "prevention": (
            "Monitor young leaves and shoots regularly and remove severely "
            "affected material when appropriate."
        )
    },

    "Healthy": {
        "severity": "None",
        "affected": "No obvious affected part detected",
        "cause": "The model classified the uploaded image as healthy.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": (
            "Continue proper watering, routine inspection and balanced "
            "nutrition."
        )
    },

    "Powdery Mildew": {
        "severity": "High",
        "affected": "Young leaves, shoots, flowers and panicles",
        "cause": "Fungal infection that can produce a white powdery growth.",
        "treatment": (
            "Remove severely affected material where appropriate. A suitable "
            "fungicide may be considered after confirming the disease."
        ),
        "prevention": (
            "Maintain good air circulation, monitor new growth and avoid "
            "excessive nitrogen fertilization."
        )
    },

    "Sooty Mould": {
        "severity": "Medium",
        "affected": "Leaf surfaces, twigs and other plant surfaces",
        "cause": (
            "Dark fungal growth commonly associated with honeydew from "
            "sap-sucking insects."
        ),
        "treatment": (
            "Identify and manage the underlying sap-sucking insects. "
            "Clean affected surfaces gently where practical."
        ),
        "prevention": (
            "Monitor for scale insects, mealybugs, aphids and other "
            "honeydew-producing pests."
        )
    }
}

# ==================================================
# MEDICINE / PEST MANAGEMENT RECOMMENDATIONS
# ==================================================

medicine_info = {
    "Anthracnose": [
        (
            "Copper oxychloride — a fungicide option mentioned in some "
            "crop-disease management guidance. Confirm the product label "
            "allows use on mango for the diagnosed disease."
        ),
        (
            "Mancozeb — a fungicide option only if currently registered "
            "and labelled for the specific mango use in your area."
        )
    ],

    "Bacterial Canker": [
        (
            "Do not assume that a general fungicide will control a "
            "bacterial disease."
        ),
        (
            "Confirm the diagnosis with a local agricultural officer and "
            "use only a treatment specifically recommended and registered "
            "for the confirmed problem."
        )
    ],

    "Cutting Weevil": [
        (
            "First confirm the weevil species and the extent of infestation."
        ),
        (
            "Use an insecticide only if its current label specifically "
            "permits the confirmed pest on mango."
        )
    ],

    "Die Back": [
        (
            "Copper oxychloride may be considered for certain causes of "
            "die-back where recommended and registered for mango."
        ),
        (
            "If symptoms continue, get the cause confirmed before choosing "
            "another fungicide or treatment."
        )
    ],

    "Gall Midge": [
        (
            "Confirm the gall midge species and the plant part being damaged."
        ),
        (
            "Select an insect-control product only when its current label "
            "covers the confirmed pest and mango crop."
        )
    ],

    "Healthy": [
        "No medicine or pesticide is recommended for a healthy-looking leaf.",
        "Continue monitoring and treat only when a pest or disease is confirmed."
    ],

    "Powdery Mildew": [
        (
            "Wettable sulphur — a fungicide option where the product label "
            "allows use on mango for powdery mildew."
        ),
        (
            "Hexaconazole — consider only if a product is currently "
            "registered and labelled for the confirmed mango disease."
        )
    ],

    "Sooty Mould": [
        (
            "Prioritize control of the sap-sucking insects responsible for "
            "honeydew; identify the pest before selecting a product."
        ),
        (
            "A copper-based treatment may be mentioned in some guidance, "
            "but confirm its suitability and current mango label before use."
        )
    ]
}

# ==================================================
# FERTILIZER RECOMMENDATIONS — TWO PER CLASS
# ==================================================

fertilizer_info = {
    "Anthracnose": [
        (
            "NPK 19:19:19 — balanced water-soluble fertilizer for plant "
            "nutrition when required."
        ),
        (
            "Well-decomposed farmyard manure (FYM) — supports soil structure "
            "and general plant nutrition."
        )
    ],

    "Bacterial Canker": [
        (
            "NPK 19:19:19 — use only when appropriate for the plant's "
            "nutritional needs."
        ),
        (
            "Well-decomposed farmyard manure (FYM) — supports soil health "
            "and general nutrition."
        )
    ],

    "Cutting Weevil": [
        (
            "NPK 19:19:19 — balanced nutrients to support healthy growth "
            "when needed."
        ),
        (
            "Neem cake — an organic soil amendment; it is not a substitute "
            "for confirming and managing a weevil infestation."
        )
    ],

    "Die Back": [
        (
            "NPK 19:19:19 — apply only if suitable for the tree's growth "
            "stage and nutrient requirements."
        ),
        (
            "Well-decomposed farmyard manure (FYM) — supports general "
            "soil fertility."
        )
    ],

    "Gall Midge": [
        (
            "NPK 19:19:19 — balanced nutrition when required; it does not "
            "control gall midge."
        ),
        (
            "Neem cake — an organic soil amendment, not a replacement for "
            "appropriate pest management."
        )
    ],

    "Healthy": [
        (
            "NPK 19:19:19 — a balanced fertilizer option when supported "
            "by the tree's nutritional needs."
        ),
        (
            "Well-decomposed farmyard manure (FYM) — supports soil health "
            "and general nutrition."
        )
    ],

    "Powdery Mildew": [
        (
            "NPK 19:19:19 — use only as needed; avoid excessive nitrogen "
            "fertilization."
        ),
        (
            "Well-decomposed farmyard manure (FYM) — supports general "
            "soil fertility."
        )
    ],

    "Sooty Mould": [
        (
            "NPK 19:19:19 — balanced nutrition when required; it does not "
            "remove sooty mould or control the insects causing it."
        ),
        (
            "Neem cake — an organic soil amendment; manage the underlying "
            "sap-sucking insects separately."
        )
    ]
}

# ==================================================
# OPENCV: HIGHLIGHT ABNORMAL SPOTS
# ==================================================

def highlight_spots(image):
    img = np.array(image.convert("RGB"))

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    difference = blurred.astype(np.int16) - gray.astype(np.int16)

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

    contours, _ = cv2.findContours(
        spots,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    number_of_spots = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if 15 <= area <= 800:
            number_of_spots += 1

            cv2.drawContours(
                result,
                [contour],
                -1,
                (255, 0, 0),
                2
            )

            x, y, w, h = cv2.boundingRect(contour)

            cv2.circle(
                result,
                (x + w // 2, y + h // 2),
                3,
                (255, 0, 0),
                -1
            )

    return result, number_of_spots

# ==================================================
# OPEN-METEO WEATHER
# ==================================================

def get_weather(city):
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_response = requests.get(
            geo_url,
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            },
            timeout=10
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return None

        location = geo_data["results"][0]

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_response = requests.get(
            weather_url,
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

        return weather_response.json().get("current")

    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None

# ==================================================
# APP INTERFACE
# ==================================================

st.title("🥭 Mango Leaf Disease Detection")

st.write(
    "Upload a mango leaf image to detect its likely disease, "
    "view care recommendations and check current weather."
)

# ==================================================
# IMAGE UPLOAD AND CNN PREDICTION
# ==================================================

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

        # Keep the preprocessing consistent with the existing app.
        resized = image.resize((224, 224))
        img_array = np.array(resized)
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analysing the leaf image..."):
            model = load_model()
            prediction = model.predict(img_array, verbose=0)

        if prediction.ndim != 2 or prediction.shape[1] != len(class_names):
            st.error(
                "The model output does not match the eight configured "
                "disease classes. Check the model and class order."
            )
            st.stop()

        predicted_index = int(np.argmax(prediction[0]))
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction[0]) * 100)

        info = disease_info[predicted_class]

        # ==========================================
        # PREDICTION RESULT
        # ==========================================

        st.success(f"🌿 Predicted Class: {predicted_class}")
        st.info(f"🎯 Model Confidence: {confidence:.2f}%")

        st.caption(
            "This is a model prediction, not a laboratory-confirmed diagnosis. "
            "Confidence is the model's output score, not a guarantee of accuracy."
        )

        # ==========================================
        # DISEASE INFORMATION
        # ==========================================

        st.header("🩺 Disease Information")

        st.write(f"**Severity:** {info['severity']}")
        st.write(f"**Affected plant parts:** {info['affected']}")
        st.write(f"**Possible cause:** {info['cause']}")
        st.write(f"**Suggested care:** {info['treatment']}")
        st.write(f"**Prevention:** {info['prevention']}")

        # ==========================================
        # MEDICINE RECOMMENDATIONS
        # ==========================================

        st.header("💊 Medicine / Pest Management")

        for medicine in medicine_info[predicted_class]:
            st.write(f"• {medicine}")

        st.caption(
            "These are educational suggestions, not a prescription. "
            "Before applying any pesticide, confirm the diagnosis and check "
            "that the exact product is currently registered and labelled for "
            "mango and the confirmed pest or disease in your area. Follow the "
            "label, protective-equipment instructions and pre-harvest interval."
        )

        # ==========================================
        # FERTILIZER RECOMMENDATIONS
        # ==========================================

        st.header("🌱 Fertilizer Recommendations")

        for index, fertilizer in enumerate(
            fertilizer_info[predicted_class],
            start=1
        ):
            st.write(f"**{index}.** {fertilizer}")

        st.caption(
            "Fertilizers support plant nutrition; they do not cure plant "
            "diseases or replace pest control. The correct fertilizer and "
            "quantity depend on tree age, soil testing and local agricultural "
            "guidance. Do not apply a fixed dose based only on the image."
        )

        # ==========================================
        # OPENCV SPOT HIGHLIGHTING
        # ==========================================

        st.header("🔍 Abnormal Spot Highlighting")

        highlighted_image, spot_count = highlight_spots(image)

        st.image(
            highlighted_image,
            caption="Red outlines indicate image regions flagged by OpenCV",
            use_container_width=True
        )

        if spot_count > 0:
            st.success(
                f"OpenCV highlighted approximately {spot_count} "
                "abnormal spot(s)."
            )
        else:
            st.warning(
                "No clear individual spots were highlighted. This does not "
                "rule out disease."
            )

        st.caption(
            "OpenCV highlights darker, spot-like regions using image "
            "processing. These regions are not a validated disease "
            "segmentation mask and may include non-disease marks."
        )

    except Exception as error:
        st.error(
            "The image could not be analysed. Check that the model is "
            "available and the uploaded image is valid."
        )
        st.exception(error)

# ==================================================
# WEATHER SECTION
# ==================================================

st.header("🌦️ Current Weather")

city = st.text_input(
    "Enter your city for current weather",
    value="Bengaluru"
)

if st.button("Check Weather"):
    if not city.strip():
        st.warning("Please enter a city name.")
    else:
        with st.spinner("Getting current weather..."):
            weather = get_weather(city.strip())

        if weather is not None:
            temperature = weather.get("temperature_2m")
            humidity = weather.get("relative_humidity_2m")
            precipitation = weather.get("precipitation")

            st.write(f"**City:** {city.strip()}")
            st.write(f"🌡️ **Temperature:** {temperature} °C")
            st.write(f"💧 **Relative humidity:** {humidity}%")
            st.write(f"🌧️ **Current precipitation:** {precipitation} mm")

            if humidity is not None and humidity >= 80:
                st.warning(
                    "Humidity is high. Monitor the mango plant for fungal "
                    "symptoms and maintain good air circulation."
                )
            elif precipitation is not None and precipitation > 0:
                st.warning(
                    "Precipitation is currently reported. Avoid unnecessary "
                    "leaf wetting and monitor plant conditions."
                )
            else:
                st.success(
                    "Continue routine plant monitoring and follow local "
                    "weather and agricultural advice."
                )
        else:
            st.error(
                "Weather information could not be retrieved. Check the city "
                "name and internet connection, then try again."
            )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Mango Leaf Disease Detection using CNN | Major Project"
)
