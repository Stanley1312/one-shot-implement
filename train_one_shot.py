# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
import sys
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
# %matplotlib inline
import cv2
import time
# import tensorflow 
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate, Layer
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
import numpy.random as rng
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-p","--pickle_dir",required=True,help="path to the pickle folder")
ap.add_argument("-e","--eval_every",default=200,type=int,help="interval for evaluating on one-shot tasks")
ap.add_argument("-b","--batch_size",default=32,type=int,help="batch size for training")
ap.add_argument("-i","--n_iter",default=200000,type=int,help="No. of training iterations")
ap.add_argument("-N","--N_way",default=20,type=int,help="how many classes for testing one-shot tasks")
ap.add_argument("-n","--n_val",default=250,type=int,help="how many one-shot tasks to validate on")
ap.add_argument("-s","--save_path",required=True,help="path to save the model file")
args = vars(ap.parse_args())

""" Load dataset from pickle file"""
for pkl in os.listdir(args["pickle_dir"]): 
    if "train" in pkl:
        with open(r"E:\Lo\Lo\one-shot-implement\train-new.pickle", "rb") as f:
            (Xtrain, train_classes) = pickle.load(f)
    if "val" in pkl:
        with open(r"E:\Lo\Lo\one-shot-implement\val-new.pickle", "rb") as f:
            (Xval, val_classes) = pickle.load(f)


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,kernel_initializer=initializers.RandomNormal(stddev=0.01,mean=0.0),kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu',
                     kernel_initializer=initializers.RandomNormal(stddev=0.01,mean=0.0),
                     bias_initializer=initializers.RandomNormal(stddev=0.01,mean=0.5), kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01,mean=0.0),
                     bias_initializer=initializers.RandomNormal(stddev=0.01,mean=0.5), kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01,mean=0.0),
                     bias_initializer=initializers.RandomNormal(stddev=0.01,mean=0.5), kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initializers.RandomNormal(stddev=0.01,mean=0.0),bias_initializer=initializers.RandomNormal(stddev=0.01,mean=0.5)))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initializers.RandomNormal(stddev=0.01,mean=0.5))(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net

model = get_siamese_model((105, 105, 1))
model.summary()

optimizer = Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer)

def get_batch(batch_size,s="train"):
    """Create batch of n pairs, half same class, half different class"""
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape
    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes,size=(batch_size,),replace=False)
    # initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)]
    # initialize vector for the targets
    targets=np.zeros((batch_size,))
    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = rng.randint(0, n_examples)
        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category  
        else: 
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1,n_classes)) % n_classes
        
        pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h,1)
    
    return pairs, targets

def make_oneshot_task(N, s="val", language=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape
    indices = rng.randint(0, n_examples,size=(N,))
    if language is not None: # if language is specified, select characters for that language
        low, high = categories[language]
        if N > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low,high),size=(N,),replace=False)
    else: # if no language specified just pick a bunch of random letters
        categories = rng.choice(range(n_classes),size=(N,),replace=False)            
    true_category = categories[0]
    #get two randome index for true category
    ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))
    #create N test image of true categories
    test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, w, h,1)
    #create N support set from random categories and indices
    support_set = X[categories,indices,:,:]
    #set the first image of support set is the true image
    support_set[0,:,:] = X[true_category,ex2,:,:]
    support_set = support_set.reshape(N, w, h,1)
    #create N targets
    targets = np.zeros((N,))
    #set the first target is true
    targets[0] = 1
    #shuffle the index of 3 arrays
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image,support_set]

    return pairs, targets

def test_oneshot(model, N, k, s = "val", verbose = 0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N,s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct

# Hyper parameters
evaluate_every = args["eval_every"] # interval for evaluating on one-shot tasks
batch_size = args["batch_size"]
n_iter = args["n_iter"] # No. of training iterations
N_way =args["N_way"]  # how many classes for testing one-shot tasks
n_val =args["n_val"] # how many one-shot tasks to validate on
best = -1

print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter+1):
    (inputs,targets) = get_batch(batch_size)
    loss = model.train_on_batch(inputs, targets)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss)) 
        val_acc = test_oneshot(model, N_way, n_val, verbose=True)
        if val_acc > 96:
          model.save_weights(os.path.join(args["save_path"], 'weights.{}-{}.h5'.format(i,val_acc)))
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc






