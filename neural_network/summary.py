
import tensorflow as tf
import math


# Summary Writer --------------------------------------------------------------
class NetworkSummary(object):
    
    def __init__(self, logdir, **file_writer_kwargs): 
        self.logdir = logdir
        
        if self.logdir is not None:
            self.writer = tf.summary.FileWriter(logdir, **file_writer_kwargs)
        else:
            self.writer = None
        
        # structure summaries
        self.layer_summaries = []

        # input/output summaries
        self.output_summary = None
        self.input_summary = None

        # training summaries
        self.loss_summary = None
        self.variable_summary = None
        
        # evaluation summaries
        self.eval_summaries = dict()
        self.per_class_eval_summaries = dict()
        
        self.train_summary = None


    def add_layer_summary(self, layer_name, layer_tensor, scope = None):

        ignore_types = ('pool', 'flatten')

        if any([(check in layer_name) for check in ignore_types]):
            # skip this layer
            return
       
        with tf.name_scope(scope, default_name = layer_name) as layer_scope:
            sparsity = tf.summary.scalar('sparsity', tf.nn.zero_fraction(layer_tensor))
            activations = tf.summary.histogram('activations', layer_tensor)
            
            layer_summaries = [sparsity, activations]
            if 'conv' in layer_name and len(self.layer_summaries) <= 0:
                kernel_images = make_kernel_images(layer_tensor)
                layer_summaries.append(kernel_images)
            
            self.layer_summaries.append(tf.summary.merge(layer_summaries))

    def add_output_summary(self, net_output, scope = None):
        with tf.name_scope(scope, default_name = 'output'):
            sparsity = tf.summary.scalar('sparsity', tf.nn.zero_fraction(net_output))
            activations = tf.summary.histogram('activations', net_output)
            self.output_summary = tf.summary.merge([sparsity, activations])
    
    def add_input_summary(self, net_input, num_samples, scope = None):
        self.input_summary = tf.summary.image('input', net_input, num_samples)        
    
    def add_loss_summary(self, total_loss_tensor, grad_loss_tensor = None, 
                         reg_penalty_tensor = None, scope = None):

        with tf.name_scope(scope, default_name = 'loss'):
            loss_summaries = [tf.summary.scalar('total_loss', total_loss_tensor)]
            
            if grad_loss_tensor is not None:
                loss_summaries.append(tf.summary.scalar('grad_loss', grad_loss_tensor))

            if reg_penalty_tensor is not None:
                loss_summaries.append(tf.summary.scalar('reg_penalty', reg_penalty_tensor))

            self.loss_summary = tf.summary.merge(loss_summaries)

    def add_eval_summary(self, eval_tensor, name = 'eval', scope = None):
        with tf.name_scope(scope, default_name = 'evaluation'):
            self.eval_summaries[name] = tf.summary.scalar(name, eval_tensor)


    def add_per_class_eval_summary(self, per_class_eval_tensor, max_val = None, 
                                   name = 'eval', scope = None):
        with tf.name_scope(scope, default_name = 'evaluation'):
            per_class_histogram_vals = make_per_class_histogram(per_class_eval_tensor, max_val = max_val)
            self.per_class_eval_summaries[name] = tf.summary.histogram(name, per_class_histogram_vals)


    def add_variable_summary(self):
        var_summaries = [tf.summary.histogram(var.op.name, var) \
                            for var in tf.trainable_variables()]

        self.variable_summary = tf.summary.merge(var_summaries)
    
    def add_graph(self, graph):
        if self.writer is not None:
            self.writer.add_graph(graph)
    
    # Getting Summary Tensors -------------------------------------------------
    
    def get_training_summary(self):
        if self.writer is None: return None
        if self.train_summary is not None: return self.train_summary

        training_summaries = self.layer_summaries

        def add_summary(summary):
            if summary is not None: 
                training_summaries.append(summary)
        
        add_summary(self.output_summary)
        add_summary(self.input_summary)
        add_summary(self.loss_summary)
        add_summary(self.variable_summary)
        
        self.train_summary = tf.summary.merge(training_summaries)

        return self.train_summary 

    def get_evaluation_summary(self, name = 'eval'):
        if self.writer is None: return None

        eval_summaries = []
        try:
            eval_summaries.append(self.eval_summaries[name])
        except KeyError:
            pass

        try:
            eval_summaries.append(self.per_class_eval_summaries[name])
        except KeyError:
            pass
        
        if len(eval_summaries) > 0:
            return tf.summary.merge(eval_summaries)
        else:
            return None

    # Writing Summaries -------------------------------------------------------

    def write(self, summary, step = None):
        if self.writer is not None:
            self.writer.add_summary(summary, step)

    def flush(self):
        if self.writer is not None: 
            self.writer.flush()


# Summary Utils ---------------------------------------------------------------

def make_per_class_histogram(per_class_results, n_digits = 4, max_val = None, scope = None):
    with tf.name_scope(scope, default_name = 'per_class_histogram_values'):
        histogram_values = []

        if max_val is not None:
            base_values = approx_decimal_for_hist(max_val, -1, n_digits)
            #print("Class -1 : {}".format(base_values.get_shape()))
            histogram_values.append(base_values)

        num_classes = per_class_results.get_shape().as_list()[0]
        
        for class_idx in range(num_classes):
            class_result = per_class_results[class_idx]
            class_values = approx_decimal_for_hist(class_result, class_idx, n_digits)
            
            #print("Class {:2d} : {}".format(class_idx, class_values.get_shape()))
            histogram_values.append(class_values)
        
    
    if tf.__version__.startswith('1.0'):
        return tf.cast(tf.concat(histogram_values, 0), tf.int32, name = 'per_class_eval_histogram')
    else:
        return tf.cast(tf.concat(0, histogram_values), tf.int32, name = 'per_class_eval_histogram')


def approx_decimal_for_hist(dec_val, fill_val, n_digits = 4):
    with tf.name_scope('hist_decimal_approx'):
        n_vals = tf.cast(dec_val * 10**n_digits, tf.int32)
        return tf.fill((n_vals,), fill_val) 


def summarize_layer(layer_tensor):
    sparsity = tf.summary.scalar('sparsity', tf.nn.zero_fraction(layer_tensor))
    activations = tf.summary.histogram('activations', layer_tensor)
    return tf.summary.merge([sparsity, activations])


def make_kernel_images(conv_tensor, tensor_scope = None):
    #print("Making convolutinal kernel images for layer '{}'".format(conv_tensor.name))

    if tensor_scope is None:
        tensor_scope = conv_tensor.name.split('/')[0]

    #print("Tensor Scope : {}".format(tensor_scope))
    #print("Trainable Variables in Scope")
    scope_vars = [var for var in tf.trainable_variables() if tensor_scope in var.name]
    #for var in scope_vars:
    #    print("  {}".format(var.name))
    
    found_weights = [var for var in scope_vars if 'weight' in var.name]

    if len(found_weights) > 1:
        raise Exception("Ambiguous weights to use as kernel image for the given tensor!")

    if len(found_weights) <= 0:
        raise Exception("No weights to use as kernel image for the given tensor!")
    
    kernel = found_weights[0]
    
    #print("Kernel Shape : {}".format(kernel.get_shape()))

    kernel_grid = put_kernels_on_grid(found_weights[0])
    
    #print("Grid Shape : {}".format(kernel_grid.get_shape()))

    return tf.summary.image('kernels', kernel_grid, max_outputs=1)


def put_kernels_on_grid(kernel, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].

    Source:
        https://gist.github.com/kukuruza/03731dc494603ceab0c5
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(math.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    #print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))
    
    # scaling to [0, 255] is not necessary for tensorboard
    return x7

