import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense

# Constants
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 64, 64, 64  # Adjust dimensions as needed
NUM_CHANNELS = 3  # Assuming RGB images
BATCH_SIZE = 16
EPOCHS = 10
CALORIES_PER_500G = 360

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = image / 255.0  # Normalize pixel values
    return image

# Load biryani volume estimation model
volume_model = Sequential([
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, NUM_CHANNELS)),
    MaxPooling3D(pool_size=(2, 2, 2)),
    Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1)  # Output layer for volume estimation
])
volume_model.compile(optimizer='adam', loss='mean_squared_error')
volume_model.load_weights("biryani_volume_estimation_model_weights.h5")

# Load biryani classification model
biryani_model = load_model("biryani_classification_model.h5")

# Function to classify image as biryani or not
def is_biryani(image_path):
    # Preprocess image
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict class probabilities
    probabilities = biryani_model.predict(image)[0]

    # Class index with the highest probability
    predicted_class = np.argmax(probabilities)

    return predicted_class == 1, probabilities[1]

# Function to estimate volume from image
def estimate_volume(image_path):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return volume_model.predict(image)[0][0]

# Function to calculate nutritional values based on estimated volume
def calculate_nutritional_values(volume):
    calories = (volume / 500) * CALORIES_PER_500G
    return {
        "calories": calories,
        # Add other nutritional values here...
    }

# Example usage
dataset_dir = r'C:\Users\rahul\Desktop\TDL_Project\Dataset\Biryani'  # Update with your dataset directory path
new_image_path = os.path.join(r'C:\Users\rahul\Desktop\TDL_Project\Dataset', 'test.jpg')  # Update with your new image path
is_biryani, biryani_probability = is_biryani(new_image_path)

if is_biryani:
    print("The image is biryani with probability:", biryani_probability)
    estimated_volume = estimate_volume(new_image_path)
    nutritional_values = calculate_nutritional_values(estimated_volume)
    print("Nutritional Values:", nutritional_values)
else:
    print("The image is not biryani with probability:", 1 - biryani_probability)
