import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error, structural_similarity
from skimage.io import imread
from os import listdir
from os.path import isfile, join
import re

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')  # Handles files without numbers by putting them at the end

def load_images_from_folder(folder):
    images = []
    filenames = [f for f in listdir(folder) if isfile(join(folder, f))]
    filenames.sort(key=extract_number)  # Sort using the custom key function
    for filename in filenames:
        img = imread(join(folder, filename))
        if img is not None:
            # Check if the image is grayscale and convert it to RGB
            if len(img.shape) == 2:  # Grayscale image
                img = np.stack((img,)*3, axis=-1)  # Convert to 3-channel
            images.append(img)
    return images, filenames

original_images_dir = 'images'
altered_images_dir = 'results_10'

original_images, original_filenames = load_images_from_folder(original_images_dir)
altered_images, altered_filenames = load_images_from_folder(altered_images_dir)

assert len(original_images) == len(altered_images), "Image count mismatch between folders"

results_df = pd.DataFrame(columns=['Filename', 'MSE', 'SSIM'])

for orig_img, alt_img, filename in zip(original_images, altered_images, original_filenames):
    mse = mean_squared_error(orig_img, alt_img)
    ssim = structural_similarity(orig_img, alt_img, multichannel=True)
    results_df = results_df.append({'Filename': filename, 'MSE': mse, 'SSIM': ssim}, ignore_index=True)

results_df.to_excel('resultsPM_10.xlsx', index=False)
