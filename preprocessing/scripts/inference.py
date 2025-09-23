import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Enable unsafe deserialization to load Lambda layer
keras.config.enable_unsafe_deserialization()

# Paths
model_path = "artifacts_removal.keras"
input_folder = ""
output_folder = ""
os.makedirs(output_folder, exist_ok=True)

# Load the model
model = tf.keras.models.load_model(model_path, compile=False)
print("Model loaded successfully!")

# Image preprocessing function
def load_and_preprocess(image_path, target_size=(256,256)):
    img = cv2.imread(image_path)  # BGR
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

# Batch inference for 34 images
for i in range(1, 35):
    input_image_path = os.path.join(input_folder, f"{i}.jpg")
    output_image_path = os.path.join(output_folder, f"{i}.png")

    # Load and preprocess
    input_img = load_and_preprocess(input_image_path)
    input_batch = np.expand_dims(input_img, axis=0)

    # Predict
    predicted_output = model.predict(input_batch)[0]
    predicted_output = np.clip(predicted_output, 0, 1)

    # Save output
    pred_uint8 = (predicted_output * 255).astype(np.uint8)
    cv2.imwrite(output_image_path, cv2.cvtColor(pred_uint8, cv2.COLOR_RGB2BGR))
    print(f" Saved: {output_image_path}")

print(f"\nAll images processed and saved to '{output_folder}'")

