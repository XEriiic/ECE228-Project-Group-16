import random
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as TF
from torchvision import transforms
import torch
def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)
def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)
def random_color_tf(img):
    color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
    imgs_tf = []
    tf = transforms.ColorJitter(
        color_jitter.brightness,
        color_jitter.contrast,
        color_jitter.saturation,
        color_jitter.hue)
    imgs_tf.append(tf(img))
    imgs = Image.fromarray(np.array(imgs_tf))
    imgs.save('C:\\Users\\Eric\\Desktop\\train_386_.png')
    return imgs

img = Image.open('C:\\Users\\Eric\\Desktop\\train_386.png')

imgs = random_color_tf(img)

#imgs.show()