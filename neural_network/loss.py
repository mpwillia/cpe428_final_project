
import tensorflow as tf


def softmax_cross_entropy_with_logits(net_output, exp_output):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = net_output,
                                                                  labels = exp_output),
                          name = 'softmax_cross_entropy_with_logits')

