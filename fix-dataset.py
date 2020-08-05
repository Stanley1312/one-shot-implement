import os 
from os import path
import shutil
import random
import numpy.random as rng
from img_augment import load_batch_augment
import imageio
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--data_dir",required=True,help="path to the dataset")
ap.add_argument("-n","--new_path",help="path to the new folder for Arabic dataset")
args = vars(ap.parse_args())

"""dataset path for fixing and augmenting data such as train and validation path in dataset folder"""
dataset_path = args["data_dir"]
""" new path for sorting the Arabic dataset"""
new_path = args["new_path"]



def fix_data(dataset_path):
    """ Seperate the Arabic Dataset follow the label name"""
    for train in os.listdir(dataset_path):
        list_file = []
        filename = train.split('.')[0]
        list_file.append(filename)
        label = filename.split('_')[3]
        dir_path = os.path.join(dataset_path,"character_"+label)
        if path.exists(dir_path):
            print("Folder is exists")
        else:
            os.mkdir(dir_path)
        for img in list_file:
            old_path = os.path.join(dataset_path,img+'.png')
            if path.exists(os.path.join(dir_path,img+".png")):
                print("Image is exists")
            else:
                shutil.move(old_path,dir_path)
def prepare_augmentation(dataset_path):
    """ Guarantee the image of each class of dataset are 20 images"""
    for alphabet in os.listdir(path):
        newalpha_path = os.path.join(new_path,alphabet)
        if os.path.exists(newalpha_path):
            print("----Aphabetpath is exists----------")
        else:
            os.mkdir(newalpha_path)
        alphabet_path = os.path.join(path,alphabet)
        for letter in os.listdir(alphabet_path):
            newlet_path = os.path.join(newalpha_path,letter)
            if os.path.exists(newlet_path):
                print("-----Letterpath is exists------")
            else:
                os.mkdir(newlet_path)
            letter_path = os.path.join(alphabet_path,letter)            
            list_images = os.listdir(letter_path)
            len_images = len(list_images)
            if len_images > 20:
                indinces = rng.choice(len_images,size=(20,),replace=False)
                for index in range(len_images):
                    if index not in indinces:
                        shutil.move(os.path.join(letter_path,list_images[index]),newlet_path)
def augment_data(dataset_path):
    """ Function for augmenting dataset after prepare the dataset"""
    print("[INFO]: Augmenting data...............")
    for alphabet in os.listdir(dataset_path):
        alphabet_path = os.path.join(dataset_path,alphabet)
        for letter in os.listdir(alphabet_path):
            letter_path = os.path.join(alphabet_path,letter)            
            augmented_data = load_batch_augment(letter_path)
            for i, augmented in enumerate(augmented_data):
                imageio.imwrite(os.path.join(letter_path,"%d.jpg" % (i,)),augmented)
    print("[INFO]:Done augmenting data...............")

augment_data(dataset_path)
if new_path:
    prepare_augmentation(dataset_path)

