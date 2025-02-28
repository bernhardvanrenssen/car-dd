import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("car_model.keras")

# Function to load and preprocess an image
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img, img_array

# Example: specify a path to a test image
img_path = "data/COCO/processed/test/tire flat/000012.jpg"
img, img_array = preprocess_image(img_path)

# Make prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]

# Define the mapping from indices to damage types
class_mapping = {0: "crack", 1: "dent", 2: "glass shatter", 3: "lamp broken", 4: "scratch", 5: "tire flat"}
print("Predicted class:", class_mapping.get(predicted_class, "Unknown"))

# Display the image with prediction
plt.imshow(img)
plt.title(f"Prediction: {class_mapping.get(predicted_class, 'Unknown')}")
plt.axis("off")
plt.show()
