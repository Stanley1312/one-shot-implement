# one-shot-implement

This project hosts the code for implementing the FCOS algorithm for object detection, as presented in this blog:

https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d

## Dataset

The Arabic Dataset: https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d

The Omnilog Dataset: https://github.com/brendenlake/omniglot

To make the architecture of Arabic dataset similar to the Omnilog dataset, use [Seperate_arabic_dataset](fix-dataset.py) function and the function [prepare_augmentation](fix-dataset.py) for balancing the amout of images in each class.

![Alt text](https://github.com/Stanley1312/one-shot-implement/blob/master/dataset.PNG?raw=true "Dataset Structure")


