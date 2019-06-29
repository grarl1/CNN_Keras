import os
import logging
import numpy as np

import noise

from PIL import Image

def load_image(path):
    '''Load image from path. 
    :param path: image file path
    :return: normalized image
    :rtype: numpy.array
    '''
    image = Image.open(path)
    image = np.array(image)
    return normalize_image(image)

def save_image(name, img, dir_path):
    '''Saves a normalized image to a directory path. Creates the directory if it does not exists.
    :param name: filename of saved image file
    :param img: numpy.array
    :param dir_path: output directory path
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Save image
    output = denormalize_image(img)
    output = Image.fromarray(output)
    output.save(os.path.join(dir_path, name))
   
def normalize_image(image):
    '''Normalize image to range [0,1]
    :param image: image
    :type image: np.array
    :return: image
    :rtype: np.array
    '''
    return image * (1.0 / 255.0)

def denormalize_image(image):
    '''Denormalize image from range [0,1] to range [0,255]
    :param image: image
    :type image: np.array
    :return: image
    :rtype: np.array
    '''
    output = image * 255.0
    return output.astype(np.uint8)

def downsample_normalized(image, config):
    '''Downsample an image using bicubic interpolation
    :param image: image
    :type image: np.array
    :return: image
    :rtype: np.array
    '''
    # Get scale
    scale = config.getint("fsrcnn", "upscale")

    # Denormalize
    output = denormalize_image(image)

    # numpy -> image
    if output.shape[2] == 1:
        output = output[:,:,0] 
    output = Image.fromarray(output)

    # Check size
    (width, height) = output.size
    if width % scale != 0 or height % scale != 0:
        raise ValueError("Only exact downscale is implemented")

    # Rescale
    output = output.resize((width//scale, height//scale), Image.LANCZOS)

    # image -> numpy
    output = np.array(output)
    if len(output.shape) == 2:
        output = output.reshape((*output.shape, 1))

    # Normalize
    output = normalize_image(output)

    return output

def preprocess_label(image, config):
    '''Preprocess label
    :param image: np.array
    :param config: config parser
    :return: Image preprocessed
    '''
    # Read config
    model_name = config.get("default", "target_net")
    scale = config.getint("fsrcnn", "upscale")

    if model_name == "FSRCNN":
        # Denormalize
        output = denormalize_image(image)

        # numpy -> image
        if output.shape[2] == 1:
            output = output[:,:,0] 
        output = Image.fromarray(output)

        # Preprocess
        width, height = output.size
        output = output.crop((0, 0, width - width % scale, height - height % scale))
        output = output.convert('YCbCr')  

        # image -> numpy
        output = np.array(output)
        output = output[:,:,0] # Keep Y channel, discard CbCr channels
        output = output.reshape((*output.shape, 1))

        # Normalize
        output = normalize_image(output)
        return output

    elif model_name == "IRCNN":
        return image
    else:
        raise ValueError("Not supported network {}".format(model_name))

def generate_data(label, config):
    '''Create data from label
    :param label: label
    :param config: config parser
    :return: data created from label
    '''
    model_name = config.get("default", "target_net")
    if model_name == "FSRCNN":
        return downsample_normalized(label, config)
    elif model_name == "IRCNN":
        return noise.add_random_noise(label)
    else:
        raise ValueError("Not supported network {}".format(model_name))

def crop_image_in_subpatches(image, crop_size, stride):
    '''Split the image in subimages. Applied padding is VALID.
    :param image: np.array
    :param crop_size: size of the crop
    :param stride: stride value
    :return: list of subpatches
    :rtype: np.array
    '''
    # Create list for subpatches
    sub_patches = []

    if len(image.shape) == 3:
        rows, cols, _ = image.shape
    else:
        rows, cols = image.shape

    for x in range(0, rows - crop_size + 1, stride):
        for y in range(0, cols - crop_size + 1, stride):
            sub_patches.append(image[x : x + crop_size, y : y + crop_size, :])
          
    return np.asarray(sub_patches)

def generate_training_data_all(data_path, config):
    '''Yields pre-processed training data
    :param data_path: training images directory path
    :param config: config parser
    :return: 2-tuples of numpy.arrays containing (data, label)
    '''
    # Read config
    target_net = config.get("default", "target_net")
    upscale = config.getint('fsrcnn', 'upscale')
    read_size = config.getint('training', 'read_size')
    batch_size = config.getint('training', 'batch_size')
    crop_size = config.getint('training', 'patch_crop_size')
    stride = config.getint('training', 'patch_stride')

    # Read image files list
    file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    
    # Sort randomly
    np.random.shuffle(file_paths)

    # Containers
    data_gen, labels_gen = [], []

    # Generate
    for i in range(0, len(file_paths), read_size):

        # Read read_size images
        data, labels = [], []
        for file_path in file_paths[i : i + read_size]:
            # Read image
            image = load_image(file_path) 
            # Preprocess label
            label = preprocess_label(image, config)
            # Crop label in subpatches
            labels_to_extend = crop_image_in_subpatches(label, crop_size, stride)
            # Generate data
            data_to_extend = [generate_data(x) for x in labels_to_extend]
            labels.extend(labels_to_extend)
            data.extend(data_to_extend)

        # Sort crops randomly
        indices = np.random.permutation(len(labels))
        labels = [labels[i] for i in indices]
        data = [data[i] for i in indices]
        
        # Generate batches
        while len(data) != 0:
            # Compute len to pop
            len_to_pop = min(len(data), batch_size - len(data_gen))

            # Pop data
            data_gen.extend(data[:len_to_pop])
            data = data[len_to_pop:]
            labels_gen.extend(labels[:len_to_pop])
            labels = labels[len_to_pop:]

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
        np.random.seed(0)
        for data in generate_training_data_all(data_path, config):
            yield data

def merge(Y, image, config):
  '''Merges super-resolved image with chroma components
  :param Y: Y chroma component to be merged
  :param image: image to extract the rest of components from
  :param config: config parser
  :return: merged image
  '''
  # Denormalize Y channel
  Y_out = denormalize_image(Y)

  # Extract CbCr from image
  cbcr = denormalize_image(image)
  cbcr = Image.fromarray(cbcr).convert("YCbCr")
  cbcr = cbcr.resize((Y.shape[1], Y.shape[0]), Image.BICUBIC)
  cbcr = np.array(cbcr)[:,:,1:]

  # Merge Y with CbCr
  output = np.concatenate((Y_out, cbcr), axis=-1)
  output = Image.fromarray(output, "YCbCr").convert("RGB")
  output = np.array(output)
  output = normalize_image(output)
  return output

def preinference(image, config):
    '''Preprocess image pre inference
    :param image: image to be preprocessed
    :param config: config parser
    :return: preprocessed image
    '''
    output = preprocess_label(image, config)
    return output.reshape((1, *output.shape))

def postinference(config, hr_image, lr_image=None):
    '''Postprocess image after inference
    :param image: image to be postprocessed
    :param config: config parser
    :return: post processed image
    '''
    model_name = config.get("default", "target_net")
    if model_name == "FSRCNN":
        output = np.clip(hr_image, 0, 1)
        return merge(output, lr_image, config)
    elif model_name == "IRCNN":
        return np.clip(hr_image, 0, 1)
    else:
        raise ValueError("Not supported network {}".format(model_name))

# This main will be used count batches in a dataset given the parameters
if __name__ == '__main__':

    # Imports specific to this main
    import argparse
    import configparser
    
    # Define arguments
    parser = argparse.ArgumentParser(description='Calculate number of steps per epoch given a specific configuration')
    parser.add_argument('-d', '--data_path', type=str, required=True, help='Data path to load the images from')
    parser.add_argument('-c', '--config', type=str, default="config.ini", help='Network configuration file')
    parser.add_argument('-o', '--output', type=str, default="preprocess_dir", help='Output directory for saved data')
    parser.add_argument('-s', '--save', action="store_true", help='Save the data in the preprocess directory')

    # Parse arguments
    args = parser.parse_args()

    # Read configuration
    config = configparser.ConfigParser()
    config.read(args.config)

    # Count batches and crops
    n_batches = 0
    n_crops = 0
    counter = 0
    for x, y in generate_training_data_all(args.data_path, config):

        if args.save:
            for x0, y0 in zip(x,y):
                save_image("x_{}.png".format(counter), x0[:,:,0], args.output)
                save_image("y_{}.png".format(counter), y0[:,:,0], args.output)
                counter += 1

        n_batches += 1
        n_crops += len(x)
        if n_batches % config.getint("training", "read_size") == 0:
            print("Counting batches: {}".format(n_batches))
            print("Counting crops: {}".format(n_crops))

    print("Final number of batches: {}".format(n_batches))
    print("Final number of crops: {}".format(n_crops))

