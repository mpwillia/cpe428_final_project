
import math
from skimage import transform
from dataset import image_util
from skimage.external.tifffile import imshow
from matplotlib import pyplot
from collections import namedtuple
import numpy as np
import random
import string

Dataset = namedtuple('Dataset', ['images', 'labels'])
Datasets = namedtuple('Datasets', ['train', 'validation', 'test'])

#dataset_files_path = "./dataset_files"

def get_label_map():
    # samples 1 through 10 are numbers '0' - '9'
    # samples 11 through 36 are uppercase letters
    # samples 37 through 62 are lowercase letters
    num_start = 1
    upper_start = 11
    lower_start = 37
   
    label_map = dict()
    char_map = dict()
    for label, char in enumerate(string.digits, start = num_start):
        label_map[label] = char
        char_map[char] = label

    for label, char in enumerate(string.ascii_uppercase, start = upper_start):
        label_map[label] = char
        char_map[char] = label
    
    for label, char in enumerate(string.ascii_lowercase, start = lower_start):
        label_map[label] = char
        char_map[char] = label
    
    null_char = '_'
    label_map[0] = null_char
    char_map[null_char] = 0

    return label_map, char_map

def make_one_hot(labels, label_map):
    max_label = max(label_map.keys()) + 1

    ohl = []
    for label in labels:
        base = np.zeros(max_label)
        base[label] = 1.0
        ohl.append(base)
    
    return ohl

def split_dataset(dataset, splits = (0.8, 0.1, 0.1)):
    
    trn_p, val_p, tst_p = splits

    ds_size = len(dataset.labels)

    num_val = int(ds_size * val_p)
    num_tst = int(ds_size * tst_p)
    num_trn = ds_size - (num_val + num_tst)
    
    zipped_dataset = list(zip(dataset.images, dataset.labels))
    random.shuffle(zipped_dataset)

    trn_data = zipped_dataset[:num_trn]
    val_data = zipped_dataset[num_trn:-num_tst]
    tst_data = zipped_dataset[-num_tst:]
    
    def unzip_data(zip_data):
        return Dataset(*tuple(zip(*zip_data)))
    
    return Datasets(unzip_data(trn_data), unzip_data(val_data), unzip_data(tst_data))

class OCRDataset(object):
    def __init__(self, dataset):
        """ 
        Arguments:
            |images_map| a map from character to a list of image paths
        """
         
        #image_paths = []
        #image_labels = []
        #for label, paths in images_map.items():
        #    #if label != 'R': continue
        #    for path in paths:
        #        image_paths.append(path)
        #        image_labels.append(label)
        #
        #print("Loading {:d} images...".format(len(image_paths)))
        #images = image_util.load_images(image_paths, as_grey = True, processes = 16)
         
        self.labeled_images = list(zip(*dataset))
        exp_data = self.expand_dataset(self.labeled_images, 
                                       rotations = [-15,15], 
                                       shears = [-15,15], 
                                       translates = [(-10,-10), (-10,10), (10,-10), (10,10)], 
                                       zooms = [10])
        self.labeled_images.extend(exp_data)

        print("After Expansion: {:d} Images".format(len(self.labeled_images)))

        #exp_data = self.expand_dataset(self.labeled_images, rotations = [-15,15], shears = [-15,15], translates = [(-10,-10), (-10,10), (10,-10), (10,10)], zooms = [10])
        #self.labeled_images.extend(exp_data)
        #print("After Expansion: {:d} Images".format(len(self.labeled_images)))

        noise_images = self.add_noise(self.labeled_images)
        self.labeled_images.extend(noise_images)
        print("After Adding Noise Variants: {:d} Images".format(len(self.labeled_images)))
        
        null_samples = self.create_null_samples(500)
        self.labeled_images.extend(null_samples)
        print("After Adding Null Samples: {:d} Images".format(len(self.labeled_images)))
        
        #print(self.labeled_images[0][0].shape)
    
    def get_label_map(self):
        return get_label_map()

    def get_dataset(self, one_hot = True):
        label_map, char_map = self.get_label_map()
        
        images, labels = zip(*self.labeled_images)
        
        mapped_labels = [char_map[label] for label in labels]
        ohl = make_one_hot(mapped_labels, label_map)
        
        dataset = Dataset(images, ohl)
        
        return split_dataset(dataset)
        #return Dataset(images, ohl)

    def create_null_samples(self, num_samples):
        
        w_sample = lambda: np.ones((64,64))
        b_sample = lambda: np.zeros((64,64))
        #wb_sample = lambda: np.concatenate(np.ones((32,64)), np.zeros((32,64)))
        #bw_sample = lambda: np.concatenate(np.zeros((32,64)), np.ones((32,64)))
       
        def two_band_sample(l_black, l_size):
            r_size = 64 - l_size
            if l_black:
                return np.concatenate((np.zeros((64,l_size)), np.ones((64,r_size))),axis=1)
            else:
                return np.concatenate((np.ones((64,l_size)), np.zeros((64,r_size))),axis=1)
        
        def rand_two_band(l_black, min_size, max_size):
            return two_band_sample(l_black, random.randint(min_size, max_size))

        wb_sample = lambda: rand_two_band(False, 8, 56)
        bw_sample = lambda: rand_two_band(True, 8, 56)

        def tri_band_sample(center_black, center_size):
            l_size = int((64 - center_size)/2.0)
            r_size = 64 - (center_size + l_size)

            if center_black:
                return np.concatenate((np.ones((64,l_size)), np.zeros((64,center_size)), np.ones((64,r_size))), axis=1)
            else:
                return np.concatenate((np.zeros((64,l_size)), np.ones((64,center_size)), np.zeros((64,r_size))), axis=1)
        
        def rand_tri_band(center_black, min_size, max_size):
            return tri_band_sample(center_black, random.randint(min_size, max_size))

        bwb_sample = lambda: rand_tri_band(False, 8, 56)
        wbw_sample = lambda: rand_tri_band(True, 8, 56)
        #bwb_sample = lambda: np.concatenate(np.zeros((16,64)), np.ones((32,64)), np.zeros(16,64))
        #wbw_sample = lambda: np.concatenate(np.ones((16,64)), np.zeros((32,64)), np.ones(16,64))
        
        sample_types = [w_sample, b_sample, wb_sample, bw_sample, bwb_sample, wbw_sample]

        print("Creating Null Samples...")
        null_samples = []
        labels = []
        for sample_num in range(num_samples):
            sample_type = sample_num % len(sample_types)
            null_samples.append(sample_types[sample_type]())
            labels.append(None)
        
        noise_null_samples = []
        def add(*args, **kwargs):
            noise_images = image_util.noise_images(null_samples, *args, **kwargs) 
            #imshow(noise_images[5])
            noise_null_samples.extend(list(zip(noise_images, labels)))


        add('gaussian')
        add('speckle')
        add('s&p', salt_vs_pepper = 0.5, amount = 0.01)
        add('gaussian', var = 0.05)
        #add('speckle', var = 0.05)
        add('speckle', var = 0.10)
        #add('s&p', salt_vs_pepper = 0.5, amount = 0.05)
        add('s&p', salt_vs_pepper = 0.5, amount = 0.25)
        add('s&p', salt_vs_pepper = 0.5, amount = 0.5)

        #pyplot.show()
        null_samples = list(zip(null_samples, labels))
        return null_samples + noise_null_samples

    def add_noise(self, dataset):
        images, labels = zip(*dataset)

        all_noise_images = []

        def add(*args, **kwargs):
            noise_images = image_util.noise_images(images, *args, **kwargs) 
            all_noise_images.extend(list(zip(noise_images, labels)))
        
        print("Creating noise images...")
        add('gaussian')
        #add('poisson')
        add('speckle')
        #add('s&p', salt_vs_pepper = 0.25, amount = 0.01)
        
        return all_noise_images
        #noise_images_a = image_util.noise_images(images, 'gaussian')

        #imshow(all_noise_images[0][0])
        #pyplot.show()

    def expand_dataset(self, dataset, 
                       rotations = [], 
                       shears = [], 
                       translates = [],
                       zooms = []):
        
        #imshow(dataset[0][0])
        images, labels = zip(*dataset)
        rotated = []
        print("Rotating images...") 
        for rotation in rotations:
            rot_images = image_util.rotate_images(images, rotation)
            #imshow(rot_images[0])
            rotated.extend(list(zip(rot_images, labels)))

        print("Shearing images...") 
        sheared = []
        for shear in shears:
            shear_images = image_util.shear_images(images, shear)
            #imshow(shear_images[0])
            sheared.extend(list(zip(shear_images, labels)))
        
        print("Translating images...") 
        translated = []
        for trans in translates:
            trans_images = image_util.translate_images(images, trans)
            #imshow(trans_images[0])
            translated.extend(list(zip(trans_images, labels)))

        print("Zooming images...") 
        zoomed = []
        for zoom_amt in zooms:
            zoom_images = image_util.zoom_images(images, zoom_amt)
            #imshow(zoom_images[0])
            zoomed.extend(list(zip(zoom_images, labels)))

        #self.labeled_images.extend(rotated)
        #self.labeled_images.extend(sheared)

        #imshow(rotated[0][0])
        #imshow(rotated[1][0])
        #imshow(sheared[0][0])
        #imshow(sheared[1][0])
    
        #pyplot.show()

        return rotated + sheared + translated + zoomed;

