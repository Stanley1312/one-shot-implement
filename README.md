# one-shot-implement

This project hosts the code for implementing the One-shot learning for Hand-written Letter classification, as presented in this blog:

https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d

## Dataset

The Arabic Dataset: https://www.kaggle.com/mloey1/ahcd1

The Omnilog Dataset: https://github.com/brendenlake/omniglot

To make the architecture of Arabic dataset similar to the Omnilog dataset, use [Seperate_arabic_dataset](fix-dataset.py) function and the function [prepare_augmentation](fix-dataset.py) for balancing the amout of images in each class.

### Dataset Structure
![Alt text](https://github.com/Stanley1312/one-shot-implement/blob/master/data.PNG?raw=true "Dataset Structure")

## Testing 
Using file [One_shot](One_shot.ipynb) for testing model

Model's weight Link : https://drive.google.com/file/d/11f8CX9hZtgaK8tiYL-ZSWxqruL3E30rh/view?usp=sharing

Pickle File link: 

- Validation data : https://drive.google.com/file/d/1zpkBcrncBC3IX5ANWRovBuaTaC9GKVHs/view?usp=sharing

- Training data : https://drive.google.com/file/d/1I3UK5G1FhpDbbGwD061vDIG1nFtSWgzO/view?usp=sharing


