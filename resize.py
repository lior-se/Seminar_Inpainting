import os
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import img_as_ubyte


def resize_images_in_folder(input_folder, output_folder, target_size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for filename in filenames:
        img = imread(os.path.join(input_folder, filename))

        img_resized = resize(img, target_size, anti_aliasing=True)

        img_resized = img_as_ubyte(img_resized)

        imsave(os.path.join(output_folder, filename), img_resized)
        print(f"Resized and saved {filename} to {output_folder}")


input_folder = 'images'
output_folder = 'images'

resize_images_in_folder(input_folder, output_folder)
