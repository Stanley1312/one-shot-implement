from imgaug import augmenters as iaa
import imgaug as ia

import imageio
import os
import numpy as np


def load_batch_augment(letter_path):
    """ Augment a single or sequence of images"""
    image_list = os.listdir(letter_path)
    images = []
    for image in image_list:
        filename = os.path.join(letter_path,image)
        read_image = imageio.imread(filename)
        images.append(read_image)
    seq = iaa.Sequential([
        iaa.Flipud(),
        iaa.AdditiveGaussianNoise(scale=(5,20))
    ])
    augmented_image = seq(images=images)
    print("---------RUN RUN RUN------------")
    return augmented_image

