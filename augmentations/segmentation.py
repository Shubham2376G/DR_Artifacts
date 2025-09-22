import cv2
import numpy as np
from matplotlib import pyplot as plt

for i in range (1,32):
    # Load the image
    image = cv2.imread(f'artifact_patches/art{i}.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Thresholding to separate artifacts
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to clean up
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Create an RGBA image (transparent background)
    b, g, r = cv2.split(image)
    alpha = clean  # Use the mask as the alpha channel
    rgba = cv2.merge([b, g, r, alpha])

    # Save the result
    cv2.imwrite(f'artifact_segment/art{i}_masked.png', rgba)

    # # Display results
    # plt.figure(figsize=(12,6))
    # plt.subplot(1,3,1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
    # plt.subplot(1,3,2), plt.imshow(clean, cmap='gray'), plt.title('Mask')
    # plt.subplot(1,3,3), plt.imshow(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)), plt.title('Masked Artifact')
    # plt.show()
