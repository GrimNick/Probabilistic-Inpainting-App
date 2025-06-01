import torch
from PIL import Image, ImageDraw
import numpy as np
import yaml
from omegaconf import OmegaConf
from torchvision import transforms
import sys
import os

# Add LaMa directory to system path
sys.path.append(os.path.abspath('D:/coding/python/coded/inPaintingProject/lama'))

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.refinement import refine_predict

# Function to load the LaMa model
def load_lama_model(config_path, checkpoint_path, device):
    with open(config_path, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    
    model = load_checkpoint(config, checkpoint_path, strict=False, map_location=device)
    model.eval()
    model.to(device)
    return model, config

# Function to perform inpainting
def inpaint_image(model, config, image, mask, device):
    # Convert image and mask to tensors
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    mask_tensor = transforms.ToTensor()(mask.convert('L')).unsqueeze(0).to(device)

    with torch.no_grad():
        # Use the refine_predict function for inpainting
        result = refine_predict(config, model, image_tensor, mask_tensor)
    
    # Convert the result back to a PIL image and return
    inpainted_image = transforms.ToPILImage()(result.squeeze(0).cpu().clamp(0, 1))
    return inpainted_image

# Function to read the removal log
def read_removal_log(log_path):
    with open(log_path, 'r') as file:
        lines = file.readlines()
    
    missing_areas = []
    for line in lines:
        area_type, coords = line.strip().split(": ")
        coords = list(map(int, coords.split(',')))
        missing_areas.append((area_type.strip(), coords))
    
    return missing_areas

# Function to create a mask based on the missing areas
def create_mask(image_size, missing_areas):
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for area_type, coords in missing_areas:
        if area_type == "rectangle":
            x, y, w, h = coords
            draw.rectangle([x, y, x + w, y + h], fill=255)
        elif area_type == "circle":
            x, y, r = coords
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
    
    return mask

# Main function
def main():
    config_path = 'big-lama/config.yaml'
    checkpoint_path = 'big-lama/models/best.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, config = load_lama_model(config_path, checkpoint_path, device)

    image_path = 'modified_test1.png'
    image = Image.open(image_path).convert('RGB')

    log_path = 'removal_log.txt'
    missing_areas = read_removal_log(log_path)

    # Create the mask based on missing areas
    mask = create_mask(image.size, missing_areas)

    # Perform inpainting
    inpainted_image = inpaint_image(model, config, image, mask, device)

    # Save the inpainted image
    inpainted_image.save('inpainted_image.png')
    print("Inpainting completed. The result is saved as 'inpainted_image.png'.")

if __name__ == '__main__':
    main()
