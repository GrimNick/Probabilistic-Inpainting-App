from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def visualize_images(image_path, mask_path, output_path):
    # Load the original image, mask, and inpainted output
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')  # 'L' mode for grayscale
    output = Image.open(output_path).convert('RGB')

    # Create a masked version of the original image
    masked_image = np.array(image)
    mask_array = np.array(mask)
    masked_image[mask_array == 255] = [0, 0, 0]  # Set masked areas to black

    # Create a figure with 1 row and 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Display the original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')  # Hide axes

    # Display the mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask Image")
    axes[1].axis('off')  # Hide axes

    # Display the masked image
    axes[2].imshow(masked_image)
    axes[2].set_title("Masked Image")
    axes[2].axis('off')  # Hide axes

    # Display the inpainted output
    axes[3].imshow(output)
    axes[3].set_title("Inpainted Output")
    axes[3].axis('off')  # Hide axes

    # Adjust layout to prevent overlap and add space between subplots
    plt.subplots_adjust(wspace=0.05)  # Adjust the width between subplots
    plt.show()

# Paths to your image, mask, and output
image_path = 'test5.jpg'
mask_path = 'mask.png'
output_path = 'output.png'

# Visualize the images
visualize_images(image_path, mask_path, output_path)
