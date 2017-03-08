
import tensorflow as tf

def accuracy(net_output, exp_output):
    correct_preds = tf.equal(tf.argmax(net_output, 1), tf.argmax(exp_output, 1))
    return tf.reduce_mean(tf.cast(correct_preds, tf.float32), name = 'accuracy')



