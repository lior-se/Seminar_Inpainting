import random
import torch
from PIL import Image
from glob import glob


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = glob('data/data_large/**/*.jpg'.format(img_root),
                              recursive=True)
            #print(glob('data/data_large/**/*.jpg'.format(img_root),
                              #recursive=True))
        else:
            self.paths = glob('data/test_large/*'.format(img_root, split))
        #print(glob('masks/*.jpg'.format(mask_root)))
        self.mask_paths = glob('masks/*.jpg'.format(mask_root)) #self.mask_paths = glob('masks/*.jpg'.format(mask_root))
        self.N_mask = len(self.mask_paths)
        #print("The current path is: ",self.paths)

    def __getitem__(self, index):
        #print("the path og getiem is: ", self.paths)
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
'''import os
from PIL import Image
from torch.utils.data import Dataset

class Places2(Dataset):
    def __init__(self, root, mask_root, img_transform, mask_transform, phase):
        super().__init__()
        self.root = root
        self.mask_root = mask_root
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.phase = phase
        self.paths = self.make_dataset(root, phase)
        self.mask_paths = self.make_dataset(mask_root, phase)
        print(f"Found {len(self.paths)} images and {len(self.mask_paths)} masks in phase {phase}")
        assert len(self.paths) == len(self.mask_paths), "Mismatch between image and mask count"

    def make_dataset(self, dir, phase):
        images = []
        dir = os.path.join(dir, phase)
        print(f"Looking for images in {dir}")
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        print(f"Found {len(images)} images in {dir}")
        return images

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp"])

    def __getitem__(self, index):
        img_path = self.paths[index]
        mask_path = self.mask_paths[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        return img, mask, img_path

    def __len__(self):
        return len(self.paths)'''
