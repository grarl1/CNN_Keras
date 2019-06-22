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
   
def crop_image_in_subpatches(image, crop_size, stride):
    '''Split the image in subimages. Applied padding is VALID.
    :param image: np.array
    :param crop_size: size of the crop
    :param stride: stride value
    :return: list of subpatches
    :rtype: list(np.array)
    '''
    # Create list for subpatches
    sub_patches = []

    rows, cols, channels = image.shape
    for x in range(0, rows - crop_size + 1, stride):
      for y in range(0, cols - crop_size + 1, stride):
        sub_patch = image[x : x + crop_size, y : y + crop_size]
        sub_patch = sub_patch.reshape([crop_size, crop_size, channels])
        sub_patches.append(sub_patch)
          
    return np.asarray(sub_patches)

def normalize_image(image):
    '''Normalize image to range [0,1]
    :param image: image
    :param image: np.array
    :return: image
    :rtype: np.array
    '''
    return image * (1.0 / 255.0)

def generate_training_data_single(data_path, config):
    '''Yields pre-processed training data
    :param data_path: training images directory path
    :param config: config parser
    :return: 2-tuples of numpy.arrays containing (data, label)
    '''
    # Read config
    read_size = config.getint('training', 'read_size')
    batch_size = config.getint('training', 'batch_size')
    crop_size = config.getint('training', 'patch_crop_size')
    stride = config.getint('training', 'patch_stride')

    # Read image list
    file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    # Sort randomly
    np.random.shuffle(file_paths)

    # Generate
    data_gen, labels_gen = [], []

    # Generate
    for i in range(0, len(file_paths), read_size):

        # Read read_size images
        labels = []
        for file_path in file_paths[i : i + read_size]:
            image = np.array(Image.open(file_path)) # Load image
            image = normalize_image(image) # Normalize image
            labels.extend(crop_image_in_subpatches(image, crop_size, stride)) # Crop in subpatches

        # Shuffle subpatches
        np.random.shuffle(labels)
        # Add random noise
        data = np.asarray([noise.add_random_noise(y) for y in labels])

        while len(data) != 0:
            # Compute len to pop
            len_to_pop = min(len(data), batch_size - len(data_gen))

            # Pop data
            data_gen.extend(data[:len_to_pop])
            labels_gen.extend(labels[:len_to_pop])
            data = np.delete(data, range(len_to_pop), 0)
            labels = np.delete(labels, range(len_to_pop), 0)

            # Yield data
            if len(data_gen) == batch_size:
                yield (np.asarray(data_gen), np.asarray(labels_gen))
                data_gen, labels_gen = [], []

    # Yield remaining data
    if len(data_gen) != 0:
        yield (np.asarray(data_gen), np.asarray(labels_gen))

def generate_training_data(data_path, config):
    '''Generates training date indefinitely
    :param data_path: training images directory path
    :param config: config parser
    :return: (data, labels) where data and labels are 4-tuples of
    (batch_size, rows, columns, channels)
    '''
    while True:
        for data in generate_training_data_single(data_path, config):
            yield data

if __name__ == '__main__':

    # Imports specific to this main
    import argparse
    import configparser
    
    # Define arguments
    parser = argparse.ArgumentParser(description='Calculate number of steps per epoch given a specific configuration')
    parser.add_argument('-d', '--data_path', type=str, required=True, help='Data path to load the images from')
    parser.add_argument('-c', '--config', type=str, default="config.ini", help='Network configuration file')

    # Parse arguments
    args = parser.parse_args()

    # Read configuration
    config = configparser.ConfigParser()
    config.read(args.config)

    n_batches = 0
    n_crops = 0
    for x, y in generate_training_data_single(args.data_path, config):
        n_batches += 1
        n_crops += len(x)
        if n_batches % config.getint("training", "read_size") == 0:
            print("Counting batches: {}".format(n_batches))
            print("Counting crops: {}".format(n_crops))

    print("Final number of batches: {}".format(n_batches))
    print("Final number of crops: {}".format(n_crops))

