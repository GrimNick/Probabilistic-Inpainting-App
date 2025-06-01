from PIL import Image, ImageDraw

def parse_removal_log(log_path):
    with open(log_path, 'r') as file:
        lines = file.readlines()

    regions = []
    for line in lines:
        parts = line.strip().split(',')
        shape_type = parts[0].strip()
        coords = tuple(map(int, parts[1:]))
        regions.append((shape_type, coords))
    return regions

def create_mask(image_size, regions):
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)

    for shape_type, coords in regions:
        if shape_type == 'rectangle' and len(coords) == 4:
            x1, y1, x2, y2 = coords
            draw.rectangle([x1, y1, x2, y2], fill=255)
        elif shape_type == 'circle' and len(coords) == 3:
            x, y, r = coords
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    return mask

# Example usage
image_path = 'modified_test1.png'
log_path = 'removal_log.txt'

image = Image.open(image_path)
regions = parse_removal_log(log_path)
mask = create_mask(image.size, regions)
mask.save('mask.png')
