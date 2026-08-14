Human Action Recognition

An AI-powered Human Action Recognition application that uses deep learning and transfer learning to classify human actions from images.

Overview

This project uses a trained deep learning model to recognize 15 different human actions from an uploaded image.

The application provides a simple Streamlit web interface where users can upload an image and receive a predicted action along with the model's confidence score.

Recognized Actions
Calling
Clapping
Cycling
Dancing
Drinking
Eating
Fighting
Hugging
Laughing
Listening to Music
Running
Sitting
Sleeping
Texting
Using a Laptop
Technologies Used
Python
TensorFlow / Keras
NumPy
Pillow
Streamlit
Deep Learning
Transfer Learning
Computer Vision
Model

The trained model is saved as human_action_model.h5.

Uploaded images are resized to 224 × 224 pixels, converted to a NumPy array, normalized to a 0–1 range, and then passed to the trained model for classification.

Application Features
Upload JPG, JPEG, or PNG images
Preview uploaded images
Predict human actions
Display prediction confidence
How to Run
Clone this repository.
Install the required dependencies using pip install -r requirements.txt.
Run the application using streamlit run app.py.
Project Purpose

This project demonstrates the practical application of deep learning and computer vision for image-based human action recognition, with the trained model deployed through a Streamlit web application.

Author

Fahad Abdullah
BS Artificial Intelligence Student
Emerson University, Multan
