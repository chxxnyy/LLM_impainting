import cv2
import numpy as np
import random
import os
import torch
from PIL import Image

def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[2] * scale))

    if img_temp.shape[0] == 1:
        img_resized = cv2.resize(img_temp[0], new_dims, interpolation=cv2.INTER_LINEAR)
        scaled_img = np.expand_dims(img_resized, axis=0)
    else:
        img_transposed = img_temp.transpose(1, 2, 0)
        scaled_img = cv2.resize(img_transposed, new_dims, interpolation=cv2.INTER_LINEAR)
        scaled_img = scaled_img.transpose(2, 0, 1)

    return scaled_img


def get_data(img, gt, scale_factor=1.3):
    scale = random.uniform(0.5, scale_factor)
    flip_p = random.uniform(0, 1)

    images = img.astype(float)
    images = scale_im(images, scale)
    images = flip(images, flip_p)
    images = images[np.newaxis, :, :, :]
    images = torch.from_numpy(images.copy()).float()

    gt = gt.astype(float)
    gt[gt == 255] = 0
    gt = flip(gt, flip_p)
    gt = scale_im(gt, scale)
    labels = gt.copy()
    return images, labels