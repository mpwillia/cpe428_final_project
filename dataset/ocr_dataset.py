
import math
from skimage import transform
from dataset import image_util
from skimage.external.tifffile import imshow
from matplotlib import pyplot

#dataset_files_path = "./dataset_files"

def get_label_map():
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
 

class OCRDataset(object):
    def __init__(self, images_map):
        """ 
        Arguments:
            |images_map| a map from character to a list of image paths
        """
         
        image_paths = []
        image_labels = []
        for label, paths in images_map.items():
            if label != 'q': continue
            for path in paths:
                image_paths.append(path)
                image_labels.append(label)
        
        print("Loading {:d} images...".format(len(image_paths)))
        images = image_util.load_images(image_paths[:100], as_grey = True, processes = 16)
        
        self.labeled_images = list(zip(images, image_labels))
        
        self.expand_dataset(rotations = [-30,30], shears = [-30,30])
        print("After Expansion: {:d} Images".format(len(self.labeled_images)))

    def expand_dataset(self, rotations = [], shears = []):
        
        images, labels = zip(*self.labeled_images)
        rotated = []
        print("Rotating images...") 
        for rotation in rotations:
            rot_images = image_util.rotate_images(images, rotation)
            rotated.extend(list(zip(rot_images, labels)))

        print("Shearing images...") 
        sheared = []
        for shear in shears:
            shear_images = image_util.shear_images(images, shear)
            sheared.extend(list(zip(shear_images, labels)))

        self.labeled_images.extend(rotated)
        self.labeled_images.extend(sheared)

        imshow(self.labeled_images[0][0])
        imshow(rotated[0][0])
        imshow(sheared[0][0])
    
        pyplot.show()

