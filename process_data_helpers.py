import os
import numpy as np
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os.path
from os import path

# importing PIL
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
from PIL import Image, ImageEnhance
import random
import time
import datetime
from p_tqdm import p_map
from random import gauss

target_directory = ""

def change_target_directory(new_direct):
    global target_directory
    target_directory = new_direct

def augment_clean_data(clean_data):
    data_list = clean_data
    print("Length of Data List: ",len(data_list))
    num_processors = 8
    print("Doing Flipped Image Data.")
    p=Pool(processes = num_processors)
    flipped_output = p_map(get_flipped_image,data_list)
    print("Doing OG Image Data.")
    clean_output = p_map(get_data_from_line,data_list)
    print("Merging Data.")

    aug = clean_output + flipped_output
    return aug

def augment_clean_data_master(master_data):
    data_list = master_data
    print("Length of Data List: ",len(data_list))
    num_processors = 8
    print("Doing Flipped Image Data.")
    p=Pool(processes = num_processors)
    flipped_output = p_map(get_data_from_line_master_flipped,data_list)
    print("Doing OG Image Data.")
    clean_output = p_map(get_data_from_line_master,data_list)
    print("Merging Data.")

    aug = clean_output + flipped_output
    return aug

def clean_dirty_data(dirty_data):
    p=Pool(processes = 8)
    legal_values = p_map(check_if_line_is_valid, dirty_data)
    return legal_values

def copy_unique_images(data_list):
    p=Pool(processes = 8)
    legal_values = p_map(check_if_line_is_unique_and_copy_image_to_master, data_list)
    return legal_values

def check_if_line_is_unique_and_copy_image_to_master(line):
    global target_directory
    target_directory = "9-15-21"
    name = line[0]
    if (path.exists("./" + "master" + "/" + name + ".jpeg")):
        print("exists.")
        print(name)
        return None
    else:
        img = cv2.imread('./' + target_directory + '/' + name + '.jpeg')
        cv2.imwrite("./master/" + name + ".jpeg", img)
        return line


def check_if_line_is_valid(item):
    global target_directory
    if len(item) == 3:
        try:
            name = item[0]
            img = get_dirty_image_from_name(name)
            if item[1] != "None":
                #save image and data to new file
                img = Image.fromarray(img)
                img.save('./' + target_directory + '/' + name + ".jpeg")
                return item
        except:
            #print("Fail 1")
            return None
    else:
        #print("Fail 2")
        return None

def get_dirty_image_from_name(name):
    try:
      img = Image.open('./data/' + name + '.jpeg') # open the image file
      img.verify() # verify that it is, in fact an image
      img = cv2.imread('./data/' + name + '.jpeg')
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      return img
    except:
      #print("Invalid")
      #print(name)
      return None


def get_image_from_name_master(name):
    img = cv2.imread('./master/' + name + '.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_image_from_name(name):
    img = cv2.imread('./clean_data/' + name + '.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_data_from_line_master(line):
    try:
        name =  line[0]#.replace(":", "_")
        img = get_image_from_name_master(name)
        #Crop_numpy_array(img, distance_from_top, distance_from_left, new_width, new_height)
        img = crop_numpy_array(img, 250 , 100, 900 ,600)
        #change_resolution(img, height, width)
        img = change_resolution(img, 66, 200)

        angle = line[1]
        #[img, line[1], line[2]] to add speed value later
        return [img,angle]
    except:
        pass

def get_data_from_line_master_flipped(line):
    try:
        name =  line[0]#.replace(":", "_")
        img = get_image_from_name_master(name)
        #Crop_numpy_array(img, distance_from_top, distance_from_left, new_width, new_height)
        img = crop_numpy_array(img, 250 , 100, 900 ,600)
        #change_resolution(img, height, width)
        img = change_resolution(img, 66, 200)

        angle = line[1]
        img = cv2.flip(img, 1)
        angle = float(angle) * -1.0
        #[img, line[1], line[2]] to add speed value later
        return [img,angle]
    except:
        pass

def get_data_from_line(line):
    try:
        name =  line[0]#.replace(":", "_")
        img = get_image_from_name(name)
        #Crop_numpy_array(img, distance_from_top, distance_from_left, new_width, new_height)
        img = crop_numpy_array(img, 250 , 100, 900 ,600)
        #change_resolution(img, height, width)
        img = change_resolution(img, 66, 200)

        angle = line[1]
        #[img, line[1], line[2]] to add speed value later
        return [img,angle]
    except:
        pass

def get_flipped_image(line):
    try:
        name =  line[0]#.replace(":", "_")
        img = get_image_from_name(name)
        #Crop_numpy_array(img, distance_from_top, distance_from_left, new_width, new_height)
        img = crop_numpy_array(img, 250 , 100, 900 ,600)
        #change_resolution(img, height, width)
        img = change_resolution(img, 66, 200)
        angle = line[1]

        img = cv2.flip(img, 1)
        angle = float(angle) * -1.0
        #[img, line[1], line[2]] to add speed value later
        return [img,angle]
    except:
        pass

def flip_image_and_angle(pair):
        img = cv2.flip(pair[0], 1)
        angle = float(pair[1]) * -1.0
        #[img, line[1], line[2]] to add speed value later
        return [img,angle]

def read_data_txt_file_into_image_data_pairs(file_name, clean=False):

    with open(file_name) as f:
        data_list = f.readlines()
        #make sublists of [angle, name, speed]
        data_list = [x.split() for x in data_list]
        f.close()

    output = []

    for line in data_list:
        image_angle_pair = get_data_from_line(line)
        #appending OG Cropped Image
        output.append(image_angle_pair)

        #flip and change brightness of images
        if clean == False:
            flipped_image_angle_pair = flip_image_and_angle(image_angle_pair)

            #append flipped image and angle
            output.append(flipped_image_angle_pair)
            #appending flipped Image w brightness change
            #output.append([change_brightness(flipped_image_angle_pair[0], gauss(0.6,1.4)),flipped_image_angle_pair[1]])
            #appending OG Image w brightness change
            #output.append([change_brightness(image_angle_pair[0], gauss(0.6,1.4)),image_angle_pair[1]])
    return output

def change_brightness(image, change_factor):
    image = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(image)
    enhanced_im = enhancer.enhance(change_factor)
    return np.asarray(enhanced_im)

def crop_numpy_array(img, distance_from_top, distance_from_left, width, height):
    img = img[distance_from_top:(distance_from_top+height), distance_from_left:(distance_from_left+width)]
    return img

def change_resolution(img_array, height, width):
    img = cv2.resize(img_array, dsize=(width, height), interpolation = cv2.INTER_AREA)
    return img

def shuffle_list(list):
    random.shuffle(list)
    return list
