import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import torch
from torchvision import transforms

import opt
from places2 import Places2
from evaluation import evaluate
from evaluation import evaluate2
from net import PConvUNet
from util.io import load_ckpt

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--snapshot', type=str, default='')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--mask_root', type=str, default='./masks')
args = parser.parse_args()

device = torch.device('cpu')
print('available ',torch.cuda.is_available())
print('current device ',torch.cuda.current_device())
print('device count ',torch.cuda.device_count())
print('device name ',torch.cuda.get_device_name(0))

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])


#dataset_val = Places2(args.root, img_transform, mask_transform, 'val')
dataset_val = Places2(args.root, args.mask_root, img_transform, mask_transform, 'val')
print('data val done')
model = PConvUNet().to(device)
print('model done')
load_ckpt(args.snapshot, [('model', model)])
print('load done')
model.eval()
print('eval done')
evaluate(model, dataset_val, device, 'results/result.jpg')
print('evaluation1 done')
evaluate2(model, dataset_val, device, 'results/res')
