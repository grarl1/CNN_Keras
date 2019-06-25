import os
import logging
import numpy as np

import noise

from PIL import Image

def load_image(path):
    '''Load image from path. 
    :param path: image file path
    :return: image
    :rtype: numpy.array
    '''
    image = Image.open(path)
    return np.array(image)

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

    # Get image
    output = denormalize_image(image)
    output = output[:,:,0] 
    output = Image.fromarray(output)

    # Check size
    (width, height) = output.size
    if width % scale != 0 or height % scale != 0:
        raise ValueError("Only exact downscale is implemented")

    # Rescale
    output = output.resize((width//scale, height//scale), Image.LANCZOS)
    output = np.array(output)
    output = np.reshape(output, (height//scale, width//scale, 1))
    return normalize_image(output)

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
        output = Image.fromarray(image) # Create PIL Image
        width, height = output.size
        output = output.crop((0, 0, width - width % scale, height - height % scale))
        output = output.convert('YCbCr') # Convert to YCbCr
        output = np.array(output) # Convert to numpy array
        output = output[:,:,0] # Keep Y channel, discard CbCr channels
        output = normalize_image(output) # Normalize image
        output = output.reshape(*output.shape, 1)
        return output

    elif model_name == "IRCNN":
        return normalize_image(image) # Normalize image

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
    :rtype: list(np.array)
    '''
    # Create list for subpatches
    sub_patches = []

    if len(image.shape) == 3:
        rows, cols, channels = image.shape
    else:
        rows, cols, channels = (*image.shape, 1)

    for x in range(0, rows - crop_size + 1, stride):
      for y in range(0, cols - crop_size + 1, stride):
        sub_patch = image[x : x + crop_size, y : y + crop_size]
        sub_patch = sub_patch.reshape([crop_size, crop_size, channels])
        sub_patches.append(sub_patch)
          
    return np.asarray(sub_patches)

def generate_training_datum(image, config):
    '''Generate a single datum from a label
    :param image: image label
    :param config: config parser
    :return: data from label
    '''
    label = preprocess_label(image, config) 
    data = generate_data(label, config)
    return (data, label)

def generate_training_data_all(data_path, config):
    '''Yields pre-processed training data
    :param data_path: training images directory path
    :param config: config parser
    :return: 2-tuples of numpy.arrays containing (data, label)
    '''
    # Read config
    target_net = config.get("default", "target_net")
    read_size = config.getint('training', 'read_size')
    batch_size = config.getint('training', 'batch_size')
    crop_size = config.getint('training', 'patch_crop_size')
    stride = config.getint('training', 'patch_stride')
    upscale = config.getint('fsrcnn', 'upscale')
    dcrop_size = (crop_size // upscale) if target_net == "FSRCNN" else crop_size
    dstride = (stride // upscale) if target_net == "FSRCNN" else stride

    # Read image list
    file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    # Sort randomly
    np.random.shuffle(file_paths)

    # Generate
    data_gen, labels_gen = [], []

    # Generate
    for i in range(0, len(file_paths), read_size):

        # Read read_size images
        labels, data = [], []
        for file_path in file_paths[i : i + read_size]:
            # Read image
            image = load_image(file_path) 
            # Generate data
            datum, label = generate_training_datum(image, config)
            # Crop image in subpatches
            labels.extend(crop_image_in_subpatches(label, crop_size, stride))
            data.extend(crop_image_in_subpatches(datum, dcrop_size, dstride))

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
  height, weight, _ = Y.shape
  Y_out = Y.reshape((height, weight, 1))
  Y_out = denormalize_image(Y_out)

  # Convert image to YCbCr
  Y_cbcr = Image.fromarray(image).convert("YCbCr")
  # Super resolve all channels
  Y_cbcr = Y_cbcr.resize((weight, height), Image.BICUBIC)
  # Get CbCr channels
  cbcr = np.array(Y_cbcr).reshape(height, weight, 3)[:,:,1:]

  # Merge Y with CbCr
  output = np.concatenate((Y_out, cbcr), axis=-1)
  output = Image.fromarray(output, "YCbCr").convert("RGB")
  return normalize_image(np.array(output))

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
        return merge(hr_image, lr_image, config)
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

    # Parse arguments
    args = parser.parse_args()

    # Read configuration
    config = configparser.ConfigParser()
    config.read(args.config)

    # Count batches and crops
    n_batches = 0
    n_crops = 0
#    counter = 0
    for x, y in generate_training_data_all(args.data_path, config):

#        for x0, y0 in zip(x,y):
#            save_image("x0_{}.png".format(counter), x0[:,:,0], "preprocess_dir")
#            save_image("y0_{}.png".format(counter), y0[:,:,0], "preprocess_dir")
#            counter += 1

        n_batches += 1
        n_crops += len(x)
        if n_batches % config.getint("training", "read_size") == 0:
            print("Counting batches: {}".format(n_batches))
            print("Counting crops: {}".format(n_crops))

    print("Final number of batches: {}".format(n_batches))
    print("Final number of crops: {}".format(n_crops))

