import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained models
food_model = tf.keras.models.load_model("models/food_classification_model.h5")
nutrition_model = tf.keras.models.load_model("models/nutrition_estimator_model.h5")

# Define food labels (should match your training data classes)
food_classes = ["Pizza", "Burger", "Salad", "Pasta", "Sushi"]

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input shape
    img = image.img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict food item
def predict_food(img):
    processed_img = preprocess_image(img)
    prediction = food_model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    return food_classes[predicted_class]

# Function to estimate nutrition
def estimate_nutrition(food_item):
    # Assume the model takes food name as input and returns nutrition values
    nutrition_input = np.array([[food_classes.index(food_item)]])  # Convert food to numerical input
    prediction = nutrition_model.predict(nutrition_input)
    return {
        "Calories": round(prediction[0][0], 2),
        "Protein": round(prediction[0][1], 2),
        "Carbs": round(prediction[0][2], 2),
        "Fats": round(prediction[0][3], 2)
    }


# Streamlit UI
st.title("🍽️ Food Nutrition Estimator")
st.write("Upload an image of your food to get estimated nutrition facts!")

# Upload food image
uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict food item
    predicted_food = predict_food(image)
    st.subheader(f"Predicted Food: {predicted_food}")

    # Estimate nutrition
    nutrition_info = estimate_nutrition(predicted_food)
    st.subheader("Estimated Nutrition Facts")
    st.write(f"**Calories:** {nutrition_info['Calories']} kcal")
    st.write(f"**Protein:** {nutrition_info['Protein']} g")
    st.write(f"**Carbohydrates:** {nutrition_info['Carbs']} g")
    st.write(f"**Fats:** {nutrition_info['Fats']} g")
