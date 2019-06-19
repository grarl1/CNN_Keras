import os
import logging
import numpy as np

import noise

from PIL import Image

def load_images(dir_path):
    '''Load images from a directory path. The directory must contain images only
    :param dir_path: directory path
    :return: list of images
    :rtype: list(np.array)
    '''
    # Store images
    images = []

    # Iterate all the files
    for file_path in os.listdir(dir_path):
        # Process only files
        file_fullpath = os.path.join(dir_path, file_path)
        if os.path.isfile(file_fullpath):
            image = Image.open(file_fullpath)
            images.append(np.array(image))

    return images

def save_image(name, img, dir_path):
    '''Saves an image to a directory path. Creates the directory if it does not exists.
    :param name: filename of saved image file
    :param img: numpy.array
    :param dir_path: output directory path
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Save image
    img = img*255
    img = img.astype(np.uint8)
    pil_img = Image.fromarray(img)
    pil_img.save(os.path.join(dir_path, name))
   
def crop_images_in_patches(images, crop_size, stride):
    '''Split the images in subimages for the training and validation. Applied padding is VALID.
    :param images: list of images
    :param images: list(np.array)
    :return: list of subpatches
    :rtype: list(np.array)
    '''
    sub_patches = []

    for image in images:
      rows, cols, channels = image.shape

      for x in range(0, rows - crop_size + 1, stride):
        for y in range(0, cols - crop_size + 1, stride):
          sub_patch = image[x : x + crop_size, y : y + crop_size]
          sub_patch = sub_patch.reshape([crop_size, crop_size, channels])
          sub_patches.append(sub_patch)
          
    return np.asarray(sub_patches)

def normalize_images(images):
    '''Normalize images to range [0,1]
    :param images: list of images
    :param images: list(np.array)
    :return: list of normalized images
    :rtype: list(np.array)
    '''
    return [image * (1.0 / 255.0) for image in images]

def load_training_data(data_path, val_data_path, config):
    '''Load training and validation images
    :param data_path: training images directory path
    :param val_path: validation images directory path
    :param config: config parser
    :return: a 2-tuple containing the list of training and validation images
    :rtype: (list(np.array), list(np.array))
    '''
    # Load images
    data = load_images(data_path)
    val_data = load_images(val_data_path)
    logging.getLogger().info("{} images loaded for training".format(len(data)))
    logging.getLogger().info("{} images loaded for validation".format(len(val_data)))

    # Read config
    crop_size = config.getint('training', 'patch_crop_size')
    stride = config.getint('training', 'patch_stride')

    # Normalize
    data = normalize_images(data)
    val_data = normalize_images(val_data)

    # Extract patches
    data = crop_images_in_patches(data, crop_size, stride)
    val_data = crop_images_in_patches(val_data, crop_size, stride)
    logging.getLogger().info("{} subpatches extracted for training".format(len(data)))
    logging.getLogger().info("{} subpatches extracted for validation".format(len(val_data)))

    logging.getLogger().info("Training subpatches shape {}".format(data.shape))
    logging.getLogger().info("Validation subpatches shape {}".format(val_data.shape))

    return (data, val_data)
 

