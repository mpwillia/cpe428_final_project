
import math
import numpy as np
from skimage import data
from skimage import transform as tf
from multiprocessing import Pool
import traceback
from functools import partial

def shear_images(images, deg_angle, processes = 8):
   proc_pool = Pool(processes = processes)
   affine = tf.AffineTransform(shear = math.radians(deg_angle), translation = (-18,4))
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

