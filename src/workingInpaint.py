from simple_lama_inpainting import SimpleLama
from PIL import Image

# Initialize the inpainting model
simple_lama = SimpleLama()

# Load your input image and mask
img_path = "modified_test1.png"
mask_path = "mask.png"

image = Image.open(img_path)
mask = Image.open(mask_path).convert('L')  # Ensure mask is in grayscale

# Perform inpainting
result = simple_lama(image, mask)

# Save the inpainted image
result.save("output.png")
