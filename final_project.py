
import tensorflow as tf
from neural_network import Network 
from neural_network import layers
from neural_network.loss import softmax_cross_entropy_with_logits
from neural_network.evaluation import accuracy
from neural_network.network_util import print_fit_results 
from neural_network import network_fileio
import numpy as np

import dataset

from skimage import data
from skimage import transform

import ocr_util

from skimage.external.tifffile import imshow
from matplotlib import pyplot

import sys
import os

import numpy as np
import cv2

#LOG_DIR="/media/mike/Main Storage/tensorflow-logs/ocr_test_logdir"
LOG_DIR=None

def main():
    print("CPE 428 Final Project")
    
    
    # 0 = train ; 1 = test ; 2 = live OCR
    task = 1

    network_name = "ocr_network"

    if task == 0:
        print("Training Network")
        ocr_dataset = dataset.chars74k.load_english()
        data = ocr_dataset.get_dataset()
        
        print("Training Size   : {:d}".format(len(data.train.labels)))
        print("Validation Size : {:d}".format(len(data.validation.labels)))
        print("Testing Size    : {:d}".format(len(data.test.labels)))

        net = load_network()
        fit_net(data, net)
        network_fileio.save(net, network_name)
    elif task == 1 or task == 2:
        print("Loading Trained Network")
        net = network_fileio.load(network_name)

        if task == 1:
            run_tests(net)
        else:
            live_ocr(net)


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

    epochs = 25
    mb_size = 256
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
            expansion = True,
            verbose = verbose)
     
    print_fit_results(fit_results, eval_fmt, 'Final Results')

def load_network(name = "ocr_network"):
    net = Network([64,64,1], 
                  [layers.convolution2d(num_outputs=32, kernel_size=7),
                   layers.max_pool2d(),
                   layers.convolution2d(num_outputs=64, kernel_size=5),
                   layers.convolution2d(num_outputs=128, kernel_size=3),
                   layers.max_pool2d(),
                   layers.flatten(),
                   #layers.fully_connected(num_outputs=1000, activation_fn=tf.nn.relu),
                   layers.fully_connected(num_outputs=1000, activation_fn=tf.nn.relu),
                   layers.fully_connected(num_outputs=63, activation_fn=None)],
                  logdir = LOG_DIR,
                  network_name = name)
    return net

def run_tests(net):
    
    test_imgs_dir = "./test_images"
    test_imgs = [
                 #("ocr_test_img1.png", "Abc2"),
                 #("ocr_test_red_boat.png", "Boat"),
                 #("ocr_test_thin_car.png", "Car"),
                 #("ocr_test_thick_car.png", "Car"),
                 #("ocr_test_font_computer_vision.png", "COMPUTERxVISION"),
                 #("ocr_test_noise.png", "Noise"),
                 #("ocr_test_noise_thick.png", "Noise"),
                 #("ocr_test_edge_blobs.png", "Blob"),
                 #("ocr_test_edge_blobs_gray.png", "Blob"),
                 ("ocr_test_i_samples.png", "iiii"),
                 ]

    #test_imgs = [
    #             ("ocr_test_thin_car.png", "Car"),
    #             ("ocr_test_noise.png", "Noise"),
    #             ]


    
    msg = "Letters: {:2d}  [Expected: {:2d}]    Word: {:15s}  [Expected: {:15s}  |  Distance: {:2d}] "
    for test_img_name, exp in test_imgs:
        test_img_path = os.path.join(test_imgs_dir, test_img_name)
        test_img = load_image(test_img_path, as_grey = True, final_size = None)

        
        if len(test_imgs) < 2:
            show_letters = False
        else:
            show_letters = False

        word, num_letters = predict_word(test_img, net, give_num_letters = True, show_letters = show_letters)
        
        dist = levenshteinDistance(word, exp)
        print(msg.format(num_letters, len(exp), word, exp, dist))


    #if net is not None:
    #    print(results)


def levenshteinDistance(s1, s2):
    """
    Source: http://stackoverflow.com/a/32558749
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def predict_word(image, net, give_num_letters = False, show_letters = False, show_filters = False):
    letter_imgs = ocr_util.segment_letters(image, show_filters = show_filters, save_images = True)
    
    if len(letter_imgs) <= 0:
        return None, 0

    letter_imgs = np.expand_dims(letter_imgs, axis = 3)
    
    if show_letters:
        for imgs in letter_imgs:    
            imshow(imgs)
        pyplot.show()

    label_map, char_map = dataset.get_label_map()
    results = net.predict(np.asarray(letter_imgs), label_map = label_map)

    word = ''.join(results)

    if give_num_letters:
        return word, len(letter_imgs)
    else:
        return word

def live_ocr(net):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,800);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600);
    
    msg = "Num Letters: {:2d}    Word: {:9s}         \r"
    show = False

    os.system('clear')
    print("")
    while True:
        # Get the frame and make it grayscale
        ret, frame = cap.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Segment the letters and perform OCR 
        if show:
            print("SHOWING LETTERS")

        word, num_letters = predict_word(frame, net, True, show_letters = show, show_filters = show)
        
        if word is None:
            word = ""

        # Report the results
        sys.stdout.write(msg.format(num_letters, word))
        sys.stdout.flush()
        
        if show: break

        # Display the resulting frame
        cv2.imshow('Camera',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            show = True
        else:
            show = False

    cap.release()
    cv2.destroyAllWindows() 

def load_image(image_path, as_grey = False, final_size = (64, 64)):
    img = data.imread(image_path, as_grey = as_grey)
    if final_size is not None:
        return transform.resize(img, final_size)
    else:
        return img

if __name__ == "__main__":
    main()
