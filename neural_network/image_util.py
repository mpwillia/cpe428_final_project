
import math
import numpy as np
from skimage import data
from skimage import color
from skimage import transform as tf
from skimage import util
from multiprocessing import Pool
import traceback
from functools import partial
import random

def to_grayscale(img):
   return color.rgb2gray(img)


def apply_random_expansion(images, processes = 8, **kwargs):
   print("\nApplying Random Expansion to Dataset")
   proc_pool = Pool(processes = processes)
   
   #print(images.shape)
   #images = np.squeeze(images, axis = 3)

   func_partial = partial(_apply_random_expansion, **kwargs)

   if processes <= 1:   
      images = [func_partial(img) for img in images]
   else: 
      images = safe_dispatch(proc_pool, func_partial, images)

   proc_pool.close()
   proc_pool.join()
   
   #images = np.expand_dims(images, axis = 3)
   return images



def _apply_random_expansion(image, rot_prob = 0.2, rot_range = (-30,30),
                                   shear_prob = 0.2, shear_range = (-30,30),
                                   trans_prob = 0.2, trans_range = ((-10,10),(-10,10)),
                                   zoom_prob = 0.2, zoom_range = (0, 15),
                                   noise_prob = 0.1):
   
   ops = []
   
   # Rotation
   if random.random() < rot_prob:
      ops.append(partial(_rotate_image, angle = random.randint(*rot_range)))
   
   # Shearing
   if random.random() < shear_prob:
      ops.append(partial(_shear_image, shear = random.randint(*shear_range)))

   # Translation
   if random.random() < trans_prob:
      ops.append(partial(_translate_image, trans = (random.randint(*trans_range[0]), random.randint(*trans_range[1]))))
   
   # Zooming
   if random.random() < zoom_prob:
      ops.append(partial(_zoom_image, zoom_amt = random.randint(*zoom_range)))

   # Noise
   if random.random() < noise_prob:
      ops.append(partial(_noise_image, noise_type = 'speckle'))

   if len(ops) <= 0:
      return image
   else:
      random.shuffle(ops)
   
   try:
      for op in ops:
         image = op(image)
   except:
      print(image.shape)
      raise
   
   return image

def _shear_image(image, shear, mode = 'edge'):
   affine = tf.AffineTransform(shear = math.radians(shear))
   return tf.warp(image, affine, mode = mode)

def _translate_image(image, trans, mode = 'edge'):
   affine = tf.AffineTransform(translation = trans)
   return tf.warp(image, affine, mode = mode)

def zoom_images(images, zoom_amt, processes = 8):
   proc_pool = Pool(processes = processes)
   
   func_partial = partial(_zoom_image, zoom_amt = zoom_amt)

   if processes <= 1:   
      images = [func_partial(img) for img in images]
   else: 
      images = safe_dispatch(proc_pool, func_partial, images)

   proc_pool.close()
   proc_pool.join()
   return images


def _zoom_image(image, zoom_amt):
   if zoom_amt <= 0: return image
   try:
      img_h, img_w = image.shape
   except:
      img_h, img_w, _ = image.shape
   scaled = tf.resize(image, (img_h + zoom_amt*2, img_w + zoom_amt*2))
   return scaled[zoom_amt:-zoom_amt, zoom_amt:-zoom_amt]


def noise_images(images, noise_type, processes = 8, **kwargs):
   proc_pool = Pool(processes = processes)
   
   func_partial = partial(_noise_image, noise_type = noise_type, **kwargs)

   if processes <= 1:   
      images = [func_partial(img) for img in images]
   else: 
      images = safe_dispatch(proc_pool, func_partial, images)

   proc_pool.close()
   proc_pool.join()
   return images

def _noise_image(image, noise_type, **kwargs):
   return util.random_noise(image, mode = noise_type, **kwargs)



def translate_images(images, trans, processes = 8):
   proc_pool = Pool(processes = processes)
   
   affine = tf.AffineTransform(translation = trans)
   func_partial = partial(_warp_image, affine = affine, mode = 'edge')
   if processes <= 1:   
      images = [func_partial(img) for img in images]
   else: 
      images = safe_dispatch(proc_pool, func_partial, images)

   proc_pool.close()
   proc_pool.join()
   return images


def shear_images(images, deg_angle, processes = 8):
   proc_pool = Pool(processes = processes)
   
   if deg_angle < 0:
      trans = (-10,5)
   else:
      trans = (10,5)

   affine = tf.AffineTransform(shear = math.radians(deg_angle), translation = trans)
   func_partial = partial(_warp_image, affine = affine, mode = 'edge')
   if processes <= 1:   
      images = [func_partial(img) for img in images]
   else: 
      images = safe_dispatch(proc_pool, func_partial, images)

   proc_pool.close()
   proc_pool.join()
   return images


def _warp_image(image, affine, mode = 'edge'):
   return tf.warp(image, affine, mode = mode)


def rotate_images(images, angle, processes = 8):
   proc_pool = Pool(processes = processes)
   func_partial = partial(_rotate_image, angle = angle, mode = 'edge')
   if processes <= 1:   
      images = [func_partial(img) for img in images]
   else: 
      images = safe_dispatch(proc_pool, func_partial, images)

   proc_pool.close()
   proc_pool.join()
   return images

def _rotate_image(image, angle, mode = 'edge'):
   return tf.rotate(image, angle, mode = mode)
   

def load_images(image_paths, as_grey = False, processes = 8, verbose = False):
   """
   Loads the given list of images
   Arguments:
      |image_paths| is the list of images to load
   Optional:
      |mode| the mode to convert image to
      |verbose| if True will print out debug information.
      |proc_pool| a process pool for this function to use, if None it will just
         run in a single process.
   
   Returns a list of images
   """
   proc_pool = Pool(processes = processes)
   load_partial = partial(_load_image, as_grey = as_grey)
   if processes <= 1:   
      images = [load_partial(path) for path in image_paths]
   else: 
      images = safe_dispatch(proc_pool, load_partial, image_paths)

   proc_pool.close()
   proc_pool.join()
   return images

def _load_image(image_path, as_grey = False, final_size = (64, 64)):
   return tf.resize(data.imread(image_path, as_grey = as_grey), final_size)

# Multiprocessing Utilities ---------------------------------------------------
def safe_dispatch(proc_pool, task_func, tasks, quiet_kill = False):
   """
   Safely dispatches the given tasks with the given function to the given
   process pool while still allowing interrupts to properly clean up the 
   active processes. This function will block until the tasks are complete.
   
   Arguments:
      |proc_pool| the process pool to use
      |function| the function to call for each task
      |tasks| the list of tasks to complete
   
   Optional:
      |quiet_kill| defaults to True, when True this will suppress the stderr in
         the worker processes. This is particularly useful in the event of a
         keyboard interrupt which will typically cause the screen to be flooded
         with tracebacks from every single child process. This is rarely useful
         however if this behavior is desired setting |quiet_kill| to False will
         leave stderr as is.
   Returns a list of the results from calling the given function for each of
      the given tasks.
   """
    
   func_partial = partial(_safe_dispatch_function_wrapper, task_func, quiet_kill)
   try:
      return proc_pool.map_async(func_partial, tasks).get(0xFFFF)
      #return proc_pool.imap(func_partial, tasks)
   except KeyboardInterrupt:
      print("\nGot Keyboard Interrupt - Terminating Process Pool")
      print("You may need to press ^C again if this hangs.")
      proc_pool.terminate() 
      proc_pool.join()
      raise

def _safe_dispatch_function_wrapper(func, quiet_kill, task):
   """
   Helper function for safe_dispatch() 
   Not intended to be called directly.
   Not defined inside safe_dispatch() as it must be in the global scope to work
   with multiprocessing.
   """
   if quiet_kill: sys.stderr = os.devnull
   try:
      return func(task)
   except KeyboardInterrupt:
      pass
   except:
      traceback.print_exc()

