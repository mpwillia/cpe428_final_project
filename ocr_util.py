
import numpy as np

from skimage import filters
from skimage import measure
from skimage.morphology import disk
from skimage import transform as tf
from skimage.external.tifffile import imshow
from skimage.io import imsave
from matplotlib import pyplot

from shutil import rmtree

import os
#OUTPUT_DIR = "./presentation_images"
OUTPUT_DIR = "./paper_images"

def get_output_path(step, name):
    name = name + "_i_test_good"
    if step is None:
        img_name = name + ".png"
    else:
        fmt = "{:02d}_{}.png"
        img_name = fmt.format(step, name)
    return os.path.join(OUTPUT_DIR, img_name)

def make_dir(path, overwrite = True):
    if os.path.exists(OUTPUT_DIR) and not os.path.isdir(OUTPUT_DIR):
        raise Exception("Expected directory path exists and is not a directory! '{}'".format(OUTPUT_DIR))

    if os.path.exists(OUTPUT_DIR) and overwrite:
        rmtree(OUTPUT_DIR)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def segment_letters(image, final_size = (64,64), show_filters = False, save_images = False):

    if save_images:
        make_dir(OUTPUT_DIR, overwrite = True)

    step = 0
    if save_images:
        imsave(get_output_path(step, 'original'), image)  
    step += 1

    # normalize the image
    filt_image = normalize_image_values(image)


    if save_images:
        imsave(get_output_path(step, 'norm_original'), filt_image)
    step += 1

    if show_filters:
        imshow(image)
    
    #filt_image = sharpen(filt_image, 1.0)
    filt_image = sharpen(filt_image, 3.0)

    if save_images:
        imsave(get_output_path(step, 'sharpened'), filt_image)
    step += 1

    #blurred_image = filters.gaussian(filt_image, sigma = 3.0)
    #filter_blurred_image = filters.gaussian(filt_image, sigma = 1.5)
    #alpha = 0.5
    #
    #filt_image = blurred_image + alpha * (blurred_image - filter_blurred_image)
    norm_image = normalize_image_values(filt_image)


    if save_images:
        imsave(get_output_path(step, 'norm_sharpened'), norm_image)
    step += 1

    if show_filters:
        imshow(norm_image)

    
    thresh = np.mean(norm_image.flatten()) * 0.7
    
    #print("Threshold : {:f}".format(thresh))
    #thresh = 0.5
    thresh_image = norm_image < thresh
    
    if show_filters:
        imshow(thresh_image)
        pyplot.show()
    
    if save_images:
        imsave(get_output_path(step, 'thresh_image'), thresh_image.astype(np.float32))
    step += 1

    cleaned_image = remove_edge_blobs(thresh_image, edge_pad = 16)
    
    if save_images:
        imsave(get_output_path(step, 'cleaned_image'), cleaned_image)
    step += 1

    if save_images:
        inv_image = 1 - norm_image
        masked_image = 1 - (inv_image * cleaned_image)
        imsave(get_output_path(step, 'masked_cleaned_image'), masked_image)
    step += 1

    if cleaned_image is None:
        return []

    crop_bounds = get_crop_bounds(cleaned_image, padding = 16)
    cropped_img = crop_image(norm_image, *crop_bounds)

    if save_images:
        imsave(get_output_path(step, 'cropped_img'), cropped_img)
    step += 1

    # now that we're just looking at the image content, lets segment
    thresh_image = cropped_img < thresh
    col_chunks = find_col_chunks(thresh_image)

    img_chunks = crop_col_chunks(cropped_img, col_chunks)
    
    if save_images:
        for num, chunk in enumerate(img_chunks):
            name = "raw_chunk{:02d}".format(num)
            imsave(get_output_path(step, name), chunk)
    step += 1

    img_chunks = [make_image_square(img) for img in img_chunks]
    
    if save_images:
        for num, chunk in enumerate(img_chunks):
            name = "square_chunk{:02d}".format(num)
            imsave(get_output_path(step, name), chunk)
    step += 1


    img_chunks = [tf.resize(img, final_size) for img in img_chunks] 
    
    if save_images:
        for num, chunk in enumerate(img_chunks):
            name = "resize_chunk{:02d}".format(num)
            imsave(get_output_path(step, name), chunk)
    step += 1

    img_chunks = [img > thresh for img in img_chunks] 


    pyplot.show()

    return img_chunks


def remove_edge_blobs(thresh_image, edge_pad = 4):
    all_labels = measure.label(thresh_image, background = 0)
    
    props = measure.regionprops(all_labels)
    
    img_h, img_w = thresh_image.shape
    
    to_keep = []
    for prop in props:
        label = prop.label
        min_row, min_col, max_row, max_col = prop.bbox
        
        if min_row <= edge_pad or min_col <= edge_pad:
            continue
            #to_remove.append(label)
        elif max_row >= img_h - edge_pad:
            #to_remove.append(label)
            continue
        elif max_col >= img_w - edge_pad:
            #to_remove.append(label)
            continue
        else:
            to_keep.append(label)
    
    if len(to_keep) <= 0:
        return None

    cleaned_image = np.zeros((all_labels.shape))
    
    for label in to_keep:
        cleaned_image += (all_labels == label)
    
    return cleaned_image
    #imshow(all_labels, cmap='spectral')
    #imshow(cleaned_image, cmap='spectral')
    #pyplot.show()

def sharpen(image, sigma):
    blurred_image = filters.gaussian(image, sigma = sigma)
    filter_blurred_image = filters.gaussian(blurred_image, sigma = sigma / 2.0)
    alpha = 0.5
    sharp_image = blurred_image + alpha * (blurred_image - filter_blurred_image)
    return normalize_image_values(sharp_image)



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
    if gap_tolerance <= 0:
        gap_tolerance = 1

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



