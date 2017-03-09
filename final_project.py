import tensorflow as tf
from neural_network import Network 
from neural_network import layers
from neural_network.loss import softmax_cross_entropy_with_logits
from neural_network.evaluation import accuracy
from neural_network.network_util import print_fit_results 
import numpy as np

import dataset

from skimage import data
from skimage import transform as tf

import ocr_util

from skimage.external.tifffile import imshow
from matplotlib import pyplot

import sys

import numpy as np
import cv2

LOG_DIR="/media/mike/Main Storage/tensorflow-logs/ocr_test_logdir"
#LOG_DIR=None

def main():
    print("CPE 428 Final Project")

    ocr_dataset = dataset.chars74k.load_english()
    
    data = ocr_dataset.get_dataset()

    net = load_network()
    fit_net(data, net)

def fit_net(data, net):

    # setup our data
    def split_dataset(dataset):
        return np.asarray(dataset.images), np.asarray(dataset.labels)

    train_data = split_dataset(data.train)
    val_data = split_dataset(data.validation)
    test_data = split_dataset(data.test)

    # Setup our training parameters
    opt = tf.train.AdamOptimizer(0.001)
    
    loss_func = softmax_cross_entropy_with_logits
    eval_func = accuracy

    epochs = 5
    mb_size = 512
    eval_freq = 5
    eval_fmt = '8.3%'
    per_class_eval = True
    sums_per_epoch = 10
    checkpoint_freq = None
    save_checkpoints = False
    verbose = True
    
    # Fit the network to our data
    fit_results = net.fit(train_data, opt, loss_func, epochs, mb_size, 
            evaluation_freq = eval_freq, evaluation_func = eval_func,
            evaluation_fmt = eval_fmt, per_class_evaluation = per_class_eval,
            validation_data = val_data, 
            test_data = test_data, 
            shuffle_freq = 1,
            l2_reg_strength = 0.0001,
            summaries_per_epoch = sums_per_epoch,
            save_checkpoints = save_checkpoints,
            checkpoint_freq = checkpoint_freq,
            verbose = verbose)
     
    print_fit_results(fit_results, eval_fmt, 'Final Results')

def load_network():
    net = Network([64,64,1], 
                  [layers.convolution2d(num_outputs=32, kernel_size=11),
                   layers.max_pool2d(),
                   layers.convolution2d(num_outputs=64, kernel_size=5),
                   layers.convolution2d(num_outputs=128, kernel_size=3),
                   layers.max_pool2d(),
                   layers.flatten(),
                   #layers.fully_connected(num_outputs=1000, activation_fn=tf.nn.relu),
                   layers.fully_connected(num_outputs=1000, activation_fn=tf.nn.relu),
                   layers.fully_connected(num_outputs=63, activation_fn=None)],
                  logdir = LOG_DIR,
                  network_name = 'ocr_network')
    return net

def run_test():
    #test_img_path = "ocr_test_img1.png"
    test_img_path = "ocr_test_red_boat.png"
    test_img = load_image(test_img_path, as_grey = True, final_size = None)

def live_ocr():
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
