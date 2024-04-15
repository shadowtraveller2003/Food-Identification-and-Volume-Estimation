import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Constants
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 64, 64, 64  # Adjust dimensions as needed
NUM_CHANNELS = 3  # Assuming RGB images
BATCH_SIZE = 16
EPOCHS = 10
CALORIES_PER_500G = {
    "biryani": 360,
    "aloo_paratha": 320
}

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = image / 255.0  # Normalize pixel values
    return image

# Load volume estimation model
volume_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1)  # Output layer for volume estimation
])
volume_model.compile(optimizer='adam', loss='mean_squared_error')
volume_model.load_weights("volume_estimation_model_weights.weights.h5")

# Load classification model
classification_model = load_model("classification_model.h5")

# Function to classify image into one of the food categories
def classify_food(image_path):
    # Preprocess image
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict class probabilities
    probabilities = classification_model.predict(image)[0]

    # Class index with the highest probability
    predicted_class_index = np.argmax(probabilities)

    # Map class index to food category
    food_categories = ["biryani", "aloo_paratha"]
    predicted_food_category = food_categories[predicted_class_index]

    return predicted_food_category, probabilities[predicted_class_index]

# Function to estimate volume from image
def estimate_volume(image_path):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return volume_model.predict(image)[0][0]

# Function to calculate nutritional values based on estimated volume and food category
def calculate_nutritional_values(volume, food_category):
    calories_per_500g = CALORIES_PER_500G[food_category]
    calories = (volume / 500) * calories_per_500g
    return {
        "calories": calories,
        # Add other nutritional values here...
    }

# Update the image file path
image_path = r'C:\Users\rahul\Desktop\TDL_Project\test.jpeg'

# Classify the image
food_category, food_probability = classify_food(image_path)


if food_category:
    print("Image is classified as", food_category, "with probability:", food_probability)
    estimated_volume = estimate_volume(image_path)
    nutritional_values = calculate_nutritional_values(estimated_volume, food_category)
    print("Nutritional Values:", nutritional_values)
else:
    print("Failed to classify the image")







