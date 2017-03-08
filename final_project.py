
import dataset

from skimage import data
from skimage import transform as tf

import ocr_util

from skimage.external.tifffile import imshow
from matplotlib import pyplot

import sys

import numpy as np
import cv2


def main():
    print("CPE 428 Final Project")

    #dataset.chars74k.load_english()

    #test_img_path = "ocr_test_img1.png"
    test_img_path = "ocr_test_red_boat.png"
    test_img = load_image(test_img_path, as_grey = True, final_size = None)
    
    cap = cv2.VideoCapture(0)

    while True:
        # Get the frame and make it grayscale
        ret, frame = cap.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Segment the letters and perform OCR 
        letter_imgs = ocr_util.segment_letters(frame)
        
        # Report the results
        msg = "Found {:2d} Letters\r".format(len(letter_imgs))
        sys.stdout.write(msg)
        sys.stdout.flush()

        # Display the resulting frame
        cv2.imshow('Camera',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 

def load_image(image_path, as_grey = False, final_size = (64, 64)):
    img = data.imread(image_path, as_grey = as_grey)
    if final_size is not None:
        return tf.resize(img, final_size)
    else:
        return img

if __name__ == "__main__":
    main()
