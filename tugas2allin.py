# Import required libraries
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from google.colab import files
from IPython.display import display, Image
import cv2
import os

# Create upload function
def upload_images():
    uploaded = files.upload()
    image_paths = {}
    for filename in uploaded.keys():
        image_paths[filename] = filename
    return image_paths

def process_leaf_image(image_path, title):
    # Read the image
    img = imageio.imread(image_path)
    
    # Ensure image is RGB if it's RGBA
    if img.shape[-1] == 4:
        img = img[..., :3]
    
    # Get individual color channels
    red_channel = img.copy()
    red_channel[:, :, 1] = 0
    red_channel[:, :, 2] = 0
    
    green_channel = img.copy()
    green_channel[:, :, 0] = 0
    green_channel[:, :, 2] = 0
    
    blue_channel = img.copy()
    blue_channel[:, :, 0] = 0
    blue_channel[:, :, 1] = 0
    
    # Convert to grayscale
    grayscale = np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    
    # Convert to binary (threshold)
    _, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    
    # Create figure with larger size
    plt.figure(figsize=(20, 12))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title(f'Original {title}', fontsize=12)
    plt.axis('off')
    
    # Red channel
    plt.subplot(2, 3, 2)
    plt.imshow(red_channel)
    plt.title('Channel Red (R)', fontsize=12)
    plt.axis('off')
    
    # Green channel
    plt.subplot(2, 3, 3)
    plt.imshow(green_channel)
    plt.title('Channel Green (G)', fontsize=12)
    plt.axis('off')
    
    # Blue channel
    plt.subplot(2, 3, 4)
    plt.imshow(blue_channel)
    plt.title('Channel Blue (B)', fontsize=12)
    plt.axis('off')
    
    # Grayscale
    plt.subplot(2, 3, 5)
    plt.imshow(grayscale, cmap='gray')
    plt.title('Grayscale', fontsize=12)
    plt.axis('off')
    
    # Binary
    plt.subplot(2, 3, 6)
    plt.imshow(binary, cmap='binary')
    plt.title('Binary (Threshold)', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print shape and data type information
    print(f"\nInformasi gambar {title}:")
    print(f"Dimensi gambar: {img.shape}")
    print(f"Tipe data: {img.dtype}")
    print(f"Nilai minimum: {img.min()}")
    print(f"Nilai maksimum: {img.max()}")

# Main execution
print("Upload gambar daun (Pepaya, Singkong, dan Kenikir):")
image_paths = upload_images()

# Process each uploaded image
for filename, path in image_paths.items():
    leaf_name = filename.split('.')[0].capitalize()
    process_leaf_image(path, leaf_name)
