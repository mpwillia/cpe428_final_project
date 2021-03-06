

import dataset

from collections import defaultdict
import string
import os
from glob import glob
import pickle
from dataset.ocr_dataset import OCRDataset
from dataset import pickle_util
from dataset import image_util

dataset_path = os.path.join(dataset.dataset_files_path, 'chars74k')
valid_dataset_types = ('Fnt','Hnd','Img')

# samples 1 through 10 are numbers '0' - '9'
# samples 11 through 36 are uppercase letters
# samples 37 through 62 are lowercase letters


def load_english(dataset_type = 'Hnd'):
    if dataset_type not in valid_dataset_types:
        raise ValueError("Invalid dataset type! Expected: {}".format(valid_dataset_types))
    
    pkl_name = get_english_pkl_name(dataset_type)
    
    if pkl_exists(pkl_name):
        print("Loading from pickle")
        dataset = pickle_util.load(pkl_name)
    else:
        print("Dataset not yet pickled!")
        print("Loading raw dataset")
        images_map = load_english_raw(dataset_type)
        dataset = load_dataset(images_map)
        pickle_util.dump(dataset, pkl_name, compress = True)
    
    return OCRDataset(dataset)

def pkl_exists(path):
    return os.path.exists(path) or os.path.exists(path + '.gz')

def get_english_pkl_name(dataset_type = 'Hnd'):
    return 'chars74k-{}.pkl.gz'.format(dataset_type)

def load_dataset(images_map):
    image_paths = []
    image_labels = []
    for label, paths in images_map.items():
        #if label != 'R': continue
        for path in paths:
            image_paths.append(path)
            image_labels.append(label)
    
    print("Loading {:d} images...".format(len(image_paths)))
    images = image_util.load_images(image_paths, as_grey = True, processes = 16)
    
    return (images, image_labels)
    

def load_english_raw(dataset_type = 'Hnd'):
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
    
    images_char_map = {label_map[label] : paths for label, paths in images_map.items()}
    return images_char_map

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

