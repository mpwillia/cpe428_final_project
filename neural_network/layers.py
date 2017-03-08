
import tensorflow as tf
import tensorflow.contrib.layers as tfcl
from functools import partial
import sys

"""
See the following for layer info

https://www.tensorflow.org/api_docs/python/contrib.layers/higher_level_ops_for_building_neural_network_layers_
"""

def _wrap_layer(layer_func, **layer_kwargs):
    layer_kwargs.pop('inputs', None)
    
    if 'scope' not in layer_kwargs:
        layer_kwargs.pop('reuse', None)

    return partial(layer_func, **layer_kwargs) 

def _handle_kwargs(given, default):
    merged = {k:v for k,v in default.items()} 
    for k,v in given.items():
        merged[k] = v
    return merged

# Specific Layer Function Wrappers --------------------------------------------

def fully_connected(**kwargs):
    fully_connected.defaults = {'activation_fn':tf.nn.sigmoid,
                                'weights_initializer':tfcl.xavier_initializer(),
                                'biases_initializer':tfcl.xavier_initializer(),
                                #'biases_initializer':tf.constant(0.1)
                                'reuse' : True
                                }

    kwargs = _handle_kwargs(kwargs, fully_connected.defaults)
    return _wrap_layer(tfcl.fully_connected, **kwargs)


def convolution2d(**kwargs):
    convolution2d.defaults = {'activation_fn':tf.nn.relu,
                              'weights_initializer':tfcl.xavier_initializer_conv2d(),
                              'biases_initializer':tfcl.xavier_initializer_conv2d(),
                              #'biases_initializer':tf.constant(0.1),
                              'stride' : 1,
                              'rate' : 1,
                              'padding' : 'SAME',
                              'reuse' : True
                              }

    kwargs = _handle_kwargs(kwargs, convolution2d.defaults)
    return _wrap_layer(tfcl.convolution2d, **kwargs)


def max_pool2d(**kwargs):
    max_pool2d.defaults = {'kernel_size' : (2,2),
                           'stride' : (2,2),
                           'padding' : 'SAME',
                            }

    kwargs = _handle_kwargs(kwargs, max_pool2d.defaults)
    return _wrap_layer(tfcl.max_pool2d, **kwargs)

# Generic Layer Function Wrappers ---------------------------------------------

_layer_funcs = [tfcl.avg_pool2d,
                tfcl.batch_norm,
                tfcl.convolution2d,
                tfcl.convolution2d_in_plane,
                tfcl.convolution2d_transpose,
                tfcl.flatten,
                tfcl.fully_connected,
                tfcl.layer_norm,
                tfcl.max_pool2d,
                tfcl.one_hot_encoding,
                tfcl.repeat,
                tfcl.safe_embedding_lookup_sparse,
                tfcl.separable_convolution2d,
                tfcl.unit_norm,
                tf.nn.dynamic_rnn,
                #tf.nn.rnn,
                ]

def _register_layer_func(layer_func):
    return partial(_wrap_layer, layer_func)

for _layer_func in _layer_funcs:
    if _layer_func.__name__ not in sys.modules[__name__].__dict__:
        #print("Registering layer_func named '{}'".format(_layer_func.__name__))
        setattr(sys.modules[__name__], _layer_func.__name__, _register_layer_func(_layer_func))



