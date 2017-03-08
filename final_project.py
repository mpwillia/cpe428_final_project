
import dataset

from skimage import data
from skimage import transform as tf

import ocr_util

def main():
    print("CPE 428 Final Project")

    #dataset.chars74k.load_english()

    #test_img_path = "ocr_test_img1.png"
    test_img_path = "ocr_test_red_boat.png"
    test_img = load_image(test_img_path, as_grey = True, final_size = None)
    
    letter_imgs = ocr_util.segment_letters(test_img)
    print("Found {:d} Letters".format(len(letter_imgs)))

def load_image(image_path, as_grey = False, final_size = (64, 64)):
    img = data.imread(image_path, as_grey = as_grey)
    if final_size is not None:
        return tf.resize(img, final_size)
    else:
        return img

if __name__ == "__main__":
    main()
