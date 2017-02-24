

import dataset

from collections import defaultdict
import string
import os
from glob import glob

dataset_path = os.path.join(dataset.dataset_files_path, 'chars74k')
valid_dataset_types = ('Fnt','Hnd','Img')

# samples 1 through 10 are numbers '0' - '9'
# samples 11 through 36 are uppercase letters
# samples 37 through 62 are lowercase letters


def load_english(dataset_type = 'Hnd'):
    if dataset_type not in valid_dataset_types:
        raise ValueError("Invalid dataset type! Expected: {}".format(valid_dataset_types))
    
    # get the list file
    list_name = "list_English_{}.m".format(dataset_type)
    dataset_list_path = os.path.join(dataset_path, 'Lists', list_name)
    
    if not os.path.exists(dataset_list_path):
        raise OSError("No list file found at '{}'".format(dataset_list_path))
    
    print("Dataset List Path : {}".format(dataset_list_path))

    # get the images directory 
    dataset_images_path = os.path.join(dataset_path, 'English', dataset_type, 'Img')
    if not os.path.exists(dataset_images_path):
        raise OSError("No file found at '{}'".format(dataset_images_path))
    if not os.path.isdir(dataset_images_path):
        raise OSError("Expected directory at  '{}'".format(dataset_images_path))

    contents = glob(os.path.join(dataset_images_path, '*'))

    print("Dataset Images Path : {}".format(dataset_images_path))
    print(contents)
    
    label_map = get_chars74k_label_map()
    print(label_map)
    
    images_map = gather_image_paths(dataset_images_path)
    
    for label in range(1, 63):
        image_paths = images_map[label]
        print("{:3d} has {:5d} images".format(label, len(image_paths)))


def get_chars74k_label_map():
    # samples 1 through 10 are numbers '0' - '9'
    # samples 11 through 36 are uppercase letters
    # samples 37 through 62 are lowercase letters
    num_start = 1
    upper_start = 11
    lower_start = 37
   
    label_map = dict()
    for label, char in enumerate(string.digits, start = num_start):
        label_map[label] = char

    for label, char in enumerate(string.ascii_uppercase, start = upper_start):
        label_map[label] = char
    
    for label, char in enumerate(string.ascii_lowercase, start = lower_start):
        label_map[label] = char
    
    return label_map
   

def gather_image_paths(images_dir, num_labels = 62):
    sample_dir_fmt = "Sample{:03d}"     
    
    images_map = dict() 
    for label in range(1,num_labels+1):
        sample_dir_name = sample_dir_fmt.format(label) 
        sample_dir_path = os.path.join(images_dir, sample_dir_name)
        sample_images = glob(os.path.join(sample_dir_path, '*.png'))
        images_map[label] = sample_images
    
    return images_map

