import sys
import numpy as np
import pickle
import os
import argparse
import cv2
import time
import numpy.random as rng

ap = argparse.ArgumentParser()
ap.add_argument("-t","--train_dir",required=True,help="path to the train dataset")
ap.add_argument("-v","--val_dir",required=True,help="path to the val dataset")
ap.add_argument("-n","--name",default="augmented",help="name of the pickle file")
ap.add_argument("-s","--save_path",required=True,help="path to the save folder for pickle file")
args = vars(ap.parse_args())


def loadimgs(path,n = 0):
    '''
    path => Path of train directory or test directory
    '''
    X=[]
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    # we load every alphabet seperately so we can isolate them later
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y,None]
        alphabet_path = os.path.join(path,alphabet)
        # every letter/category has it's own column in the array, so  load seperately
        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images=[]
            letter_path = os.path.join(alphabet_path, letter)
            # read all the images in the current category
            count = 0
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                assert len(os.listdir(letter_path)) <= 40

                image = cv2.imread(image_path,0)
                if image.shape[0] == 105:
                    pass
                else:
                    image = cv2.resize(image,(105,105))
                # print(image.shape)
                # image = tf.cast(image,tf.float32)
                image = image.astype(np.float32)
                category_images.append(image)
                y.append(curr_y)
                count += 1
            try:
                X.append(np.stack(category_images))
            # edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1
    y = np.vstack(y)
    X = np.stack(X)
    return X,y,lang_dict


Xtrain,ytrain,c = loadimgs(args["train_dir"])
Xval,yval,cval=loadimgs(args["val_dir"])

with open(os.path.join(args["save_path"],"train-{}.pickle".format(args["name"])), "wb") as f:
    pickle.dump((Xtrain,c),f)
with open(os.path.join(args["save_path"],"val-{}.pickle".format(args["name"])), "wb") as f:
    pickle.dump((Xval,cval),f)
