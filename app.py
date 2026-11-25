import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

import os
import gdown
import joblib

GESTURE_MODEL_PATH = "gesture_model.pkl"

if not os.path.exists(GESTURE_MODEL_PATH):
    file_id = "1z4IEuMnmH3QA324dRaJfgzJJnv-LExxb"

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(
        url,
        GESTURE_MODEL_PATH,
        quiet=False
    )

gesture_bundle = joblib.load(GESTURE_MODEL_PATH)
gesture_bundle = joblib.load("gesture_model.pkl")
energy_bundle = joblib.load("energy_model.pkl")

gesture_model = gesture_bundle["model"]
label_encoder = gesture_bundle["label_encoder"]
image_size = gesture_bundle["image_size"]

energy_model = energy_bundle["model"]
feature_names = energy_bundle["feature_names"]

st.title("Machine Learning Prediction System")
st.write("Demonstration of Classification and Regression Models")

tab1, tab2 = st.tabs(["Hand Gesture Recognition", "Energy Prediction"])


with tab1:

    st.header("Hand Gesture Recognition")

    uploaded_file = st.file_uploader(
        "Upload a hand gesture image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", width=300)

        processed_image = image.convert("L").resize(
            (image_size, image_size)
        )

        image_array = np.array(processed_image)

        image_array = image_array.flatten()

        image_array = image_array.astype("float32") / 255.0

        image_array = image_array.reshape(1, -1)

        prediction = gesture_model.predict(image_array)

        predicted_gesture = label_encoder.inverse_transform(
            prediction
        )[0]

        probabilities = gesture_model.predict_proba(image_array)[0]

        st.success(f"Predicted Gesture: {predicted_gesture}")

        st.subheader("Prediction Confidence")

        for i, class_name in enumerate(label_encoder.classes_):

            confidence = probabilities[i] * 100

            st.write(f"{class_name}: {confidence:.2f}%")

            st.progress(float(probabilities[i]))

with tab2:

    st.header("Appliance Energy Consumption Prediction")

    st.write(
        "Enter the environmental sensor values "
        "to predict appliance energy consumption."
    )

    input_values = {}

    for feature in feature_names:

        input_values[feature] = st.number_input(
            label=feature,
            value=0.0,
            format="%.3f"
        )

    if st.button("Predict Energy Consumption"):

        input_df = pd.DataFrame(
            [input_values],
            columns=feature_names
        )

        prediction = energy_model.predict(input_df)

        st.success(
            f"Predicted Appliance Energy Consumption: "
            f"{prediction[0]:.2f} Wh"
        )