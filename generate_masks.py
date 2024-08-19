import argparse
import numpy as np
import random
from PIL import Image
import os


# Function to draw a circle
def draw_circle(canvas, center, radius):
    y, x = np.ogrid[:canvas.shape[0], :canvas.shape[1]]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
    canvas[mask] = 0
    return canvas


# Function to draw a square
def draw_square(canvas, top_left, side_length):
    x_start, y_start = top_left
    x_end = min(x_start + side_length, canvas.shape[1])
    y_end = min(y_start + side_length, canvas.shape[0])
    canvas[y_start:y_end, x_start:x_end] = 0
    return canvas


def add_random_shape(canvas, target_area):
    img_size = canvas.shape[0]
    masked_pixels = 0

    while masked_pixels < target_area:
        shape_type = random.choice(['circle', 'square'])
        if shape_type == 'circle':
            radius = random.randint(10, 50)
            center = (random.randint(radius, img_size - radius - 1), random.randint(radius, img_size - radius - 1))
            canvas = draw_circle(canvas, center, radius)
        elif shape_type == 'square':
            side_length = random.randint(20, 100)
            top_left = (random.randint(0, img_size - side_length), random.randint(0, img_size - side_length))
            canvas = draw_square(canvas, top_left, side_length)

        masked_pixels = np.sum(canvas == 0)

    return canvas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='masks')
    parser.add_argument('--area_percentage', type=float, default=60,
                        help='Percentage of the area to be masked (0-100)')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    total_pixels = args.image_size ** 2
    target_area = int((args.area_percentage / 100) * total_pixels)

    for i in range(args.N):
        canvas = np.ones((args.image_size, args.image_size)).astype("i")
        mask = add_random_shape(canvas, target_area)
        print("Saving mask:", i, "Masked pixels:", np.sum(mask == 0))

        img = Image.fromarray((mask * 255).astype(np.uint8))
        img.save(f'{args.save_dir}/{i:06d}.jpg')
