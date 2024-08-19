import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize


def evaluate2(model, dataset, device, filename, num_images=100):
    # Initialize a counter for the total number of images processed
    total_images_processed = 0

    # Loop through the dataset in batches of 8 images
    for i in range(0, num_images, 8):
        # Retrieve a batch of images, masks, and ground truths from the dataset
        batch_size = min(8,
                         num_images - total_images_processed)  # Handle the case where the last batch has fewer than 8 images
        image, mask, gt = zip(*[dataset[j] for j in range(i, i + batch_size)])
        image = torch.stack(image)
        mask = torch.stack(mask)
        gt = torch.stack(gt)

        with torch.no_grad():
            # Pass the images and masks through the model
            output, _ = model(image.to(device), mask.to(device))

        # Move the output to CPU and unnormalize the images
        output = output.to(torch.device('cpu'))
        unnormalized_output = unnormalize(output)

        # Save each unnormalized output image individually
        for idx, img in enumerate(unnormalized_output):
            # Construct the individual file name by appending the index to the base filename
            individual_filename = f"{filename}_{total_images_processed + idx}.jpg"
            save_image(img, individual_filename)

        total_images_processed += batch_size

        # Break the loop if the required number of images is processed
        if total_images_processed >= num_images:
            break

    print(f"Saved {total_images_processed} images with the base filename '{filename}'.")
def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)

