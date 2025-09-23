import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use only GPU 3

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import numpy as np
import cv2
from tensorflow.image import ssim
import matplotlib.pyplot as plt
from tqdm import tqdm

print("GPUs visible to TensorFlow:")
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

# Paths
base_path = 'dataset'
source1_path = os.path.join(base_path, 'path_to_augmented')
full_clear_path = os.path.join(base_path, 'path_to_original')

# Image loader
def load_and_preprocess(image_path, target_size=(256,256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

# Data generator
def data_generator(batch_size=8, image_size=(256, 256), fraction=0.9):
    source1_images = sorted(os.listdir(source1_path))
    total_images = int(len(source1_images) * fraction)
    source1_images = source1_images[:total_images]
    
    while True:
        for i in range(0, len(source1_images), batch_size):
            batch_input, batch_target = [], []
            for j in range(i, min(i + batch_size, len(source1_images))):
                img1 = load_and_preprocess(os.path.join(source1_path, source1_images[j]), image_size)
                fused = load_and_preprocess(os.path.join(full_clear_path, source1_images[j]), image_size)
                batch_input.append(img1)
                batch_target.append(fused)
            yield np.array(batch_input), np.array(batch_target)

# Residual block
def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# Residual learning model with scaled residual output
def build_residual_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = residual_block(c1, 64)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = residual_block(c2, 128)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c3 = residual_block(c3, 256)

    # Bottleneck
    b = layers.Conv2D(512, (3,3), activation='relu', padding='same')(c3)
    b = residual_block(b, 512)

    # Decoder
    u3 = layers.UpSampling2D((2,2))(b)
    u3 = layers.Concatenate()([u3, c2])
    c4 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(u3)
    c4 = residual_block(c4, 256)

    u2 = layers.UpSampling2D((2,2))(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u2)
    c5 = residual_block(c5, 128)

    # Predict residual, scale small
    residual_output = layers.Conv2D(3, (1,1), activation='tanh')(c5)
    scaled_residual = layers.Lambda(lambda x: 0.1 * x)(residual_output)

    # Final output = input + residual
    outputs = layers.Add()([inputs, scaled_residual])

    return models.Model(inputs, outputs)

# Perceptual loss using VGG19
vgg = VGG19(weights="imagenet", include_top=False, input_shape=(256,256,3))
vgg.trainable = False
perceptual_layer = Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)

def perceptual_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(perceptual_layer(y_true) - perceptual_layer(y_pred)))

# SSIM loss
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))

# Combined loss: SSIM + MSE + Perceptual
def combined_loss(y_true, y_pred):
    ssim_component = ssim_loss(y_true, y_pred)
    mse_component = tf.reduce_mean(tf.square(y_true - y_pred))
    perceptual_component = perceptual_loss(y_true, y_pred)
    return 0.2 * ssim_component + 0.5 * mse_component + 0.3 * perceptual_component

# Build and compile
model = build_residual_model(input_shape=(256, 256, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
              loss=combined_loss, metrics=['mse', ssim_loss])

# Train
model.fit(data_generator(), steps_per_epoch=400, epochs=15)
model.save("artifacts_removal.keras")

# Test one image
test_filename = sorted(os.listdir(source1_path))[0]
img1 = load_and_preprocess(os.path.join(source1_path, test_filename))
fused = load_and_preprocess(os.path.join(full_clear_path, test_filename))
input_image_batch = np.expand_dims(img1, axis=0)

predicted_output = model.predict(input_image_batch)[0]
predicted_output = np.clip(predicted_output, 0, 1)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.title("Input"); plt.imshow(img1); plt.axis('off')
plt.subplot(1,3,2); plt.title("Predicted Output"); plt.imshow(predicted_output); plt.axis('off')
plt.subplot(1,3,3); plt.title("Target"); plt.imshow(fused); plt.axis('off')
plt.tight_layout()
plt.savefig("result.png")

# Generate for all images
save_dir = 'inference'
os.makedirs(save_dir, exist_ok=True)

filenames = sorted(os.listdir(source1_path))
for fname in tqdm(filenames, desc="Generating fused images"):
    img1 = load_and_preprocess(os.path.join(source1_path, fname))
    input_batch = np.expand_dims(img1, axis=0)
    pred = model.predict(input_batch)[0]
    pred = np.clip(pred, 0, 1)
    pred_uint8 = (pred * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, fname), cv2.cvtColor(pred_uint8, cv2.COLOR_RGB2BGR))

print(f"All images saved to: {save_dir}")


