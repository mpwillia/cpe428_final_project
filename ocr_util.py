
import numpy as np

from skimage import filters
from skimage.morphology import disk
from skimage import transform as tf
from skimage.external.tifffile import imshow
from matplotlib import pyplot


def segment_letters(image, final_size = (64,64)):
    
    # normalize the image
    filt_image = normalize_image_values(image)

    #imshow(image)
    blurred_image = filters.gaussian(filt_image, sigma = 3.0)
    filter_blurred_image = filters.gaussian(filt_image, sigma = 1.5)
    alpha = 0.5
    
    filt_image = blurred_image + alpha * (blurred_image - filter_blurred_image)
    norm_image = normalize_image_values(filt_image)
    #imshow(norm_image)

    
    thresh = np.mean(norm_image.flatten()) * 0.75
    
    #print("Threshold : {:f}".format(thresh))
    #thresh = 0.5
    thresh_image = norm_image < thresh
    
    #imshow(thresh_image)
    #pyplot.show()

    crop_bounds = get_crop_bounds(thresh_image, padding = 16)
    cropped_img = crop_image(norm_image, *crop_bounds)
    
    # now that we're just looking at the image content, lets segment
    thresh_image = cropped_img < thresh
    col_chunks = find_col_chunks(thresh_image)

    img_chunks = crop_col_chunks(cropped_img, col_chunks)
    img_chunks = [make_image_square(img) for img in img_chunks]
    img_chunks = [tf.resize(img, final_size) for img in img_chunks] 
    img_chunks = [img > thresh for img in img_chunks] 

    return img_chunks


def make_image_square(image):
    img_h, img_w = image.shape
    dom_size = max(image.shape) 
    
    pad_w = dom_size - img_w
    pad_wl = int(pad_w / 2.0)
    pad_wr = pad_w - pad_wl

    pad_h = dom_size - img_h
    pad_hl = int(pad_h / 2.0)
    pad_hr = pad_h - pad_hl

    #return np.pad(image, ((pad_hl, pad_hr), (pad_wl, pad_wr)), 'constant', constant_values = 1.0) 
    return np.pad(image, ((pad_hl, pad_hr), (pad_wl, pad_wr)), 'maximum') 
    
def crop_col_chunks(image, col_chunks):
    img_chunks = []
    img_h, img_w = image.shape
    for col_chunk in col_chunks:
        img_chunks.append(crop_image(image, 0, img_h, *col_chunk))
    
    return img_chunks

def find_col_chunks(image, gap_tolerance = 4, chunk_padding = 8):
    col_means = np.mean(image, axis = 0)
    nonzero_indices = np.asarray(col_means.nonzero()).flatten()
    img_h, img_w = image.shape

    chunks = []
    def add_chunk(start, end):
        start = start - chunk_padding
        if start < 0: start = 0
        
        end = end + chunk_padding
        if end > img_w: end = img_w

        chunks.append((start, end)) 

    chunk_start = None
    prev_idx = 0
    for idx in nonzero_indices:
        if chunk_start is None:
            chunk_start = idx
        elif (idx - prev_idx) > gap_tolerance:
            add_chunk(chunk_start, prev_idx)
            chunk_start = idx
            
        prev_idx = idx
    
    if len(chunks) > 0:
        add_chunk(chunk_start, prev_idx)

    return chunks

def normalize_image_values(image):
    max_val = max(image.flatten())
    min_val = min(image.flatten())
    val_range = max_val - min_val
    
    if val_range == 0.0: return image
    if val_range == 1.0: return image
    return (image - min_val) / val_range


def crop_image(image, row_start, row_end, col_start, col_end):
    return image[row_start:row_end , col_start:col_end]

def get_crop_bounds(image, padding = 0):

    # compute the means of each row and column to simplify the calc
    col_means = np.mean(image, axis = 0)
    row_means = np.mean(image, axis = 1)
    
    # find the first and last non-zero values for both row and col
    def find_start_end(vals):
        nonzero_indices = np.asarray(vals.nonzero())
        return min(nonzero_indices.flatten()), max(nonzero_indices.flatten())

    col_start, col_end = find_start_end(col_means)
    row_start, row_end = find_start_end(row_means)
    
    # now add padding
    img_h, img_w = image.shape
    col_start = col_start - padding
    if col_start < 0: col_start = 0
    
    col_end = col_end + padding
    if col_end > img_w: col_end = img_w

    row_start = row_start - padding
    if row_start < 0: row_start = 0
    
    row_end = row_end + padding
    if row_end > img_h: row_end = img_h
    
    return row_start, row_end, col_start, col_end



