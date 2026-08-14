import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("human_action_model.h5")

# Real class names
class_names = [
    "calling",
    "clapping",
    "cycling",
    "dancing",
    "drinking",
    "eating",
    "fighting",
    "hugging",
    "laughing",
    "listening_to_music",
    "running",
    "sitting",
    "sleeping",
    "texting",
    "using_laptop"
]

st.title("Human Action Recognition App")

st.write("Upload an image and the AI will predict the action.")

# Upload image
file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file is not None:

    # Open image
    image = Image.open(file)

    # Show image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize image
    img = image.resize((224, 224))

    # Convert to numpy array
    img = np.array(img)

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_action = class_names[class_index]

    # Show results
    st.success(f"Predicted Action: {predicted_action}")
    st.info(f"Confidence: {confidence:.2f}")