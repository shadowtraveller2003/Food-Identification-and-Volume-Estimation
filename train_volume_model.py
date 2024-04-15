import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Function to load volume dataset
def load_volume_dataset():
    # Load images from directory
    biryani_images = []  # List to store biryani images
    aloo_paratha_images = []  # List to store aloo paratha images
    
    # Load biryani images
    for i in range(1, 26):  # Assuming 25 images for biryani
        image_path = f'C:\\Users\\rahul\\Desktop\\TDL_Project\\Dataset\\Biryani\\biryani_{i}.jpg'
        image = cv2.imread(image_path)  # Load image using cv2.imread
        if image is not None:  # Check if the image is loaded successfully
            image = cv2.resize(image, (64, 64))  # Resize image to (64, 64)
            biryani_images.append(image)
    
    # Load aloo paratha images
    for i in range(1, 26):  # Assuming 25 images for aloo paratha
        image_path = f'C:\\Users\\rahul\\Desktop\\TDL_Project\\Dataset\\Aloo_Paratha\\aloop_{i}.jpeg'
        image = cv2.imread(image_path)  # Load image using cv2.imread
        if image is not None:  # Check if the image is loaded successfully
            image = cv2.resize(image, (64, 64))  # Resize image to (64, 64)
            aloo_paratha_images.append(image)

    # Combine images and shuffle if necessary
    images = np.concatenate([biryani_images, aloo_paratha_images], axis=0)
    
    # Generate corresponding volume labels (this is just a placeholder)
    volumes = np.random.rand(len(images)) * 1000  # Random volumes for demonstration
    
    return images, volumes

# Preprocess volume labels (if necessary)
def preprocess_volume_labels(volumes):
    return volumes

# Define model architecture for volume estimation
volume_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1)  # Output layer for volume estimation
])

# Compile the model
volume_model.compile(optimizer='adam', loss='mean_squared_error')

# Load dataset containing images and their corresponding volume labels
images, volumes = load_volume_dataset()

# Preprocess the volume labels (if necessary)
volumes = preprocess_volume_labels(volumes)

# Train the model
volume_model.fit(images, volumes, epochs=10, batch_size=16)

# Save the trained model weights for later use during inference
volume_model.save_weights("volume_estimation_model_weights.weights.h5")
