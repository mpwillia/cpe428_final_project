# CPE 428 Final Project - Handwritten OCR

This is team NLP's final project for CPE 428.
For this project we did handwritten OCR using a convolutional neural network trained on a modified version of the [Chars74k dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/).

## Character Recognition

The neural network implementation was done using [Tensorflow](https://www.tensorflow.org/) and was based off of this [Tensorflow wrapper](https://github.com/mpwillia/Tensorflow-Network-Experiments).

We made major additions to the [Tensorflow wrapper](https://github.com/mpwillia/Tensorflow-Network-Experiments) for the purposes of this project.
This includes adding the ability to randomly expand the dataset between each training iteration.

## Word Recognition

For word recognition we made extensive use of Numpy and Skimage to segment the letters out of the given image.

The source code for word recognition can be found in "ocr_util.py" which is somewhat of a misnomer. It is not so much utilities, it quickly evolved into our word recognition implementation.

