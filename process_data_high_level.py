#!/usr/bin/python3

import os
import numpy as np
import cv2
import os.path
from os import path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# importing PIL
import matplotlib.pyplot as plt
import multiprocessing as mp

from PIL import Image, ImageEnhance
import random
import datetime
import time
from multiprocessing import Pool
import h5py

from random import gauss
from sklearn import metrics
import process_data_helpers
import shutil
from distutils.dir_util import copy_tree

import keras
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Cropping2D, Activation, Flatten, Lambda, Input, ELU
from keras.callbacks import EarlyStopping

import tensorflow as tf
import shutil
print("Num of GPUs available: ", len(tf.test.gpu_device_name()))

#removes clean data destination folder with the same name
#names the folder the # of folders currently here & the date it thinks it is
def clean_conflicting_folder_and_create_empty(input_date):
    #STEP:0
    print("Cleaning environment..")
    current_date = input_date
    folder_path = "/home/oliver/Desktop/process/" + current_date
    file_path = folder_path + "/" + current_date + "_clean_data.txt"

    #make a clean_data folder
    try:
        shutil.rmtree(folder_path)
    except:
        pass

    os.mkdir(folder_path)

    #make a clean_data.txt file
    try:
        os.remove(file_path)
    except:
        pass
    f = open(folder_path + "/" + current_date + "_clean_data.txt", "w")
    f.close()
    print("Done Cleaning Environment.")

#make sure the numbe of images and lines in clean data folder match
def validate_clean_data_match(input_date):

    file_name = "./" + input_date + "/" + input_date + "_clean_data.txt"
    with open(file_name) as f:
            data_list = f.readlines()
            #make sublists of [angle, name, speed]
            data_list = [x.split(',') for x in data_list]
            f.close()

    justNames = []

    for i in data_list:
        if ((path.exists('./'+ input_date + '/' + str(i[0]) + '.jpeg')) == True):
            justNames.append(i[0])
        else:
            print("Missing Image:", i)

    print("Missing:", len(data_list)-len(justNames))
    return (len(data_list)-len(justNames))

def save_clean_images_and_angles(input_date):
    #STEP:1
    print("Opening Raw Data...")
    file_name = "data.txt"
    with open(file_name) as f:
            data_list = f.readlines()
            #make sublists of [angle, name, speed]
            data_list = [x.split(',') for x in data_list]
            f.close()

    print("Extracting Valid Data...")
    process_data_helpers.change_target_directory(input_date)
    legal_values = process_data_helpers.clean_dirty_data(data_list)

    print("Writing Clean Data to Text File...")
    count = 0
    for item in legal_values:
        if item != None:
            with open("./"+ input_date + "/" + input_date + "_clean_data.txt", "a") as output:
                    output.write(item[0] + ',' + item[1] + ',' + item[2])
                    output.close()
        else:
            count+=1
    print(count + (len(data_list)-len(legal_values)), " Pictures/Lines Cleaned Out.")

    if (validate_clean_data_match(input_date) == 0):
        print("Data Cleaning Successful.")
    else:
        cont = input("Continue? (y/n)")
        if cont == y:
            pass
        else:
            exit()

def copy_unique_images_and_values_to_master(input_date):
    #read in current date folder csv, line for line check if that image exists in the master folder
    #and if the line exists in the master data text file
    file_name = "./"+ input_date + "/" + input_date + "_clean_data.txt"
    with open(file_name) as f:
            data_list = f.readlines()
            #make sublists of [angle, name, speed]
            data_list = [x.split(',') for x in data_list]
            f.close()

    lines_to_append_to_master = process_data_helpers.copy_unique_images(data_list)
    scraggler_lines_recheck = []
    for line in lines_to_append_to_master:
        if line != None:
            with open("./"+"master"+"/"+"/master.txt", "a") as output:
                    output.write(line[0] + ',' + line[1] + ',' + line[2])

def mrMeeseeks(i,chunk_size, data_list):
    #these two indexes are always chunk_size in size apart
    start_index = i * chunk_size
    stop_index = (i+1) * chunk_size

    print("*"*30)
    print("Start Index: ", start_index/chunk_size)
    #read in the lines from the text data & pull necessary images and angles
    partial_augmented_image_data_list = process_data_helpers.augment_clean_data_master(data_list[start_index:stop_index])

    #if we're at the last chunk, then use the starting index to the end using a :
    if (i == int(len(data_list)/chunk_size)-1):
        partial_augmented_image_data_list = process_data_helpers.augment_clean_data_master(data_list[start_index:])

    #make empty lists to store images temporarily
    X = np.empty([chunk_size,66,200,3])
    y = np.empty([chunk_size,1])

    #cull the error lines and pull out the images and angles

    for count, element in enumerate(partial_augmented_image_data_list):
        try:
            if (element[1] == 'None'):
                print("FUCKED UP, KEEP THE CHECK")
                pass
            else:
                X[count] = element[0]
                y[count] = float(element[1])
        except:
            pass

    X = X.reshape(-1, 66, 200, 3)
    y = y.reshape(len(y), 1)


    #if it's the first interaction with saving the data, then you have to instantiate the files
    #if start_index == 0:
    with h5py.File('./masterArrays/Data_Chunk_' + str(i) + '.h5', 'w') as hf:
        hf.create_dataset("X", data=X, compression="gzip", chunks=True, maxshape=(50000,66, 200, 3))
        hf.create_dataset("y", data=y, compression="gzip", chunks=True, maxshape=(50000,1))

    #if we're at the last chunk, then use the starting index to the end using a :
    '''
    else:
        with h5py.File('./masterArrays/Data.h5', 'a') as hf:
            hf["X"].resize((hf["X"].shape[0] + X.shape[0]), axis = 0)
            hf["X"][-X.shape[0]:] = X

            hf["y"].resize((hf["y"].shape[0] + y.shape[0]), axis = 0)
            hf["y"][-X.shape[0]:] = y
    '''

    print("X Shape: ", X.shape)
    print("y Shape: ", y.shape)

    print(X[0])
    print(y[0])
    print("((((((((((()))))))))))")
    print(X[1])
    print(y[1])

    #clearing ram manually here cause the GC was fucking up
    del X
    del y
    del partial_augmented_image_data_list
    import gc
    gc.collect()

import multiprocessing

def augment_data_and_save_as_hdf5():
    #take master data, augment it, save in hdf5 format
    file_name = "./master/master.txt"

    #read in image name list
    with open(file_name) as f:
        data_list = f.readlines()
        #make sublists of [angle, name, speed]
        data_list = [x.split(',') for x in data_list]
        f.close()

    chunk_size = 25000

    #make a for loop that goes in increments of chunk_size & makes that data
    #& appends it to the .h5 array
    for i in range(int(len(data_list)/chunk_size)):
        #start process
        p = multiprocessing.Process(target=mrMeeseeks, args=(i,chunk_size,data_list,))
        p.start()
        p.join()
        #add a .join command every 6 iterations so we can use only 6 cores and make sure each one works then finishes.


#this method sequentially reads in the .h5 data and passes it to keras
def myReader(chunk_num):
    while True:
        #do something to define path_features

        f = h5py.File('./masterArrays/Data_Chunk_' + chunk_num + '.h5', 'r')
        X_master= f['X'][...]
        y_master = f['y'][...]
        f.close()

        x = np.load('masterArrays/Data_Chunk_' + chunk_num + '.h5')
        y = np.load(path_features + 'y_' + trainOrTest + '.npy')

        #if you're loading them already in a shape accepted by your model:
        yield (x_master,y_master)

def load_data_normalize_and_train(current_date):

    '''
    path, dirs, files = next(os.walk("./masterArrays"))
    file_count = len(files)

    print(file_count)

    #don't concat these H5 Files but make a data generator in keras
    #i think we have to combine the .h5 files first

    for n in range(int(file_count/2)):

        if n == 0:
            with h5py.File('./masterArrays/clean_master_x_'+ str(n) +'.h5', 'r') as hf:
                X_master = hf["sd_car_x"][:]

            with h5py.File('./masterArrays/clean_master_y_'+ str(n) +'.h5', 'r') as hf:
                y_master = hf["sd_car_y"][:]

        else:
            with h5py.File('./masterArrays/clean_master_x_'+ str(n) +'.h5', 'r') as hf:
                X = hf["sd_car_x"][:]
                print("Array X: ", n)
                print(X.shape)
                X_master = np.concatenate((X_master, X), axis=0)

            with h5py.File('./masterArrays/clean_master_y_'+ str(n) +'.h5', 'r') as hf:
                y = hf["sd_car_y"][:]
                y_master = np.concatenate((y_master, y), axis=0)

    '''

    f = h5py.File('./masterArrays/Data_Chunk_0.h5', 'r')
    X_master= f['X'][...]
    y_master = f['y'][...]
    f.close()

    # Start of MODEL Definition
    input_shape = (66, 200, 3)

    model = Sequential()
    # Input normalization layer
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, name='lambda_norm'))

    # 5x5 Convolutional layers with stride of 2x2
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding="valid", name='conv1', activation="elu"))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding="valid", name='conv2', activation="elu"))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding="valid", name='conv3', activation="elu"))

    # 3x3 Convolutional layers with stride of 1x1
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding="valid", name='conv4', activation="elu"))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding="valid", name='conv5', activation="elu"))
    # Flatten before passing to the fully connected layers
    model.add(Flatten())
    # Three fully connected layers
    model.add(Dense(1164, name='fc1', activation="elu"))
    model.add(Dropout(.5, name='do1'))
    model.add(Dense(100, name='fc2', activation="elu"))
    model.add(Dropout(.5, name='do2'))
    model.add(Dense(10, name='fc3', activation="elu"))
    model.add(Dropout(.5, name='do3'))
    # Output layer with tanh activation
    model.add(Dense(1, name='output'))

    adam = Adam(lr=0.001)
    model.compile(optimizer="adam", loss="mse")

    #normalize DATA
    '''
    vec = y_master.astype(np.float)
    # get max and min
    maxVec = max(vec);
    minVec = min(vec);

    # normalize to -1...1
    vecN = ((vec-minVec)/(maxVec-minVec) - 0.5 ) *2;

    # to "de-normalize", apply the calculations in reverse
    vecD = (vecN/2+0.5) * (maxVec-minVec) + minVec

    y = vecN
    '''
    y = y_master
    #START TRAINING
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=500, verbose=1, mode='auto', restore_best_weights=True)
    history = model.fit(X_master, y, batch_size=512, epochs=500, validation_split=0.20, callbacks=[monitor], verbose=1)

    #Save Model
    print("Saving ML Model")
    model.save('./models/'+current_date+'.h5')

def remove_working_data_and_file():
    print("Removing Working Data Directories...")
    shutil.rmtree("./data")
    os.remove("./data.txt")

def save_clean_data_to_zip(current_date):
    shutil.make_archive(current_date, 'zip', current_date)

def copy_dirty_data_from_sd():
    # copy subdirectory example
    print("Copying Dirty Data from SD.")
    fromDirectory = "/media/oliver/rootfs/home/pi/Desktop/data"
    os.mkdir("/home/oliver/Desktop/process/data")
    toDirectory = "/home/oliver/Desktop/process/data"

    copy_tree(fromDirectory, toDirectory)
    shutil.copy2('/media/oliver/rootfs/home/pi/Desktop/data.txt', '/home/oliver/Desktop/process/') # target filename is /dst/dir/file.ext
    print("Done Copying Dirty Data from SD.")

def clean_sd_card():
    #delete old
    print("Removing Old Data from SD Card...")
    shutil.rmtree("/media/oliver/rootfs/home/pi/Desktop/data")
    os.remove("/media/oliver/rootfs/home/pi/Desktop/data.txt")

    #make new
    print("Making Clean Destination Folders on SD Card...")
    f = open("/media/oliver/rootfs/home/pi/Desktop/data.txt", "x")
    os.mkdir("/media/oliver/rootfs/home/pi/Desktop/data")
