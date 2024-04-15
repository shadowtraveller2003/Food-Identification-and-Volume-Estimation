import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Function to load classification dataset
def load_classification_dataset():
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
    
    # Generate corresponding labels (this is just a placeholder)
    labels = np.array([0] * len(biryani_images) + [1] * len(aloo_paratha_images))  # Assuming 0 for biryani and 1 for aloo paratha
    
    return images, labels



# Preprocess labels (if necessary)
def preprocess_labels(labels):
    return labels

# Define model architecture for classification
classification_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
classification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load dataset containing images and their corresponding labels
images, labels = load_classification_dataset()

# Preprocess the labels (if necessary)
labels = preprocess_labels(labels)

# Train the model
classification_model.fit(images, labels, epochs=10, batch_size=16)

# Save the trained model for later use during inference
classification_model.save("classification_model.h5")
