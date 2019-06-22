import os
import argparse
import numpy as np
from PIL import Image

def uniform(image, noise_ratio):
    '''Applies uniform noise to an image
    :param image: numpy.array
    :param noise_ratio: Value between 0 and 1 indicating the percentage of noise to be added
    :return: Noisy image
    :rtype: numpy.array
    '''
    # Generate noise
    noise = np.random.uniform(0, 1, image.shape)

    # Apply noise
    noisy_image = image.copy()
    
    idx = np.random.rand(*noisy_image.shape) < noise_ratio
    noisy_image[idx] = noise[idx]

    return noisy_image

def gaussian(image, mean, stdev):
    '''Applies Gaussian noise to an image
    :param image: numpy.array
    :param mean: Mean for Gauss distribution
    :param stdev: Standard deviation for Gauss distribution
    :return: Noisy image
    :rtype: numpy.array
    '''
    # Generate noise
    noise = np.random.normal(mean, stdev, image.shape)

    # Apply noise
    noisy_image = np.clip(image + noise, 0, 1)

    return noisy_image

def poisson(image):
    '''Applies Poisson noise to an image
    :param image: numpy.array
    :return: Noisy image
    :rtype: numpy.array
    '''
    # Generate noise
    scale = 2**8
    noise = np.random.poisson(image * scale, image.shape)

    # Apply noise
    noisy_image = np.clip(noise / scale, 0, 1)

    return noisy_image

def salt_and_pepper(image, noise_ratio):
    '''Applies salt and pepper noise to an image
    :param image: numpy.array
    :param noise_ratio: Value between 0 and 1 indicating the percentage of noise to be added
    :return: Noisy image
    :rtype: numpy.array
    '''
    # Get dimensions
    nrows, ncols, nchannels = image.shape

    # Generate noise
    noise = np.random.randint(0, 2, (nrows, ncols, nchannels))

    # Apply noise
    noisy_image = image.copy()

    # Apply the noise to each channel
    idx = np.random.rand(nrows, ncols, nchannels) < noise_ratio
    noisy_image[idx] = noise[idx]

    return noisy_image

def add_random_noise(img):
    '''Apply random noise to the given image
    :param img: image to apply noise to
    :type img: np.array
    :return: noisy image
    :rtype: np.array
    '''
    noise_functions = [uniform, gaussian, poisson, salt_and_pepper]
    noise_function = np.random.choice(noise_functions)
    
    if noise_function == uniform:
        args = (np.random.rand() * 0.6,)
    elif noise_function == gaussian:
        args = (0, np.random.rand() * 0.5)
    elif noise_function == salt_and_pepper:
        args = (np.random.rand() * 0.25,)
    else:
        args = ()

    return noise_function(img, *args)

def add_noise_to_images(input_dir, output_dir, noise_function, noise_args):
    for item in os.listdir(input_dir):
        # Get image path
        print("Procesing {}".format(item))
        path = os.path.join(input_dir, item)

        # Only process files
        if os.path.isfile(path):

            # Open image and normalize
            image = Image.open(path)
            image = np.array(image) / 255.0
            
            # Process
            args = [image] + noise_args
            noisy_image = noise_function(*args)
            noisy_image = (noisy_image*255).astype(np.uint8)

            # Save images
            Image.fromarray(noisy_image).save(os.path.join(output_dir, item))

_noise_functions = {
    "uniform": uniform,
    "gaussian": gaussian,
    "poisson": poisson,
    "salt_pepper": salt_and_pepper,
    "random": add_random_noise,
}

if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Add noise to images in a directory')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('-n', '--noise_type', type=str, required=True, choices=_noise_functions.keys(), help='Type of noise to be added.')
    parser.add_argument('-r', '--ratio', type=float, default=0.1, help='Noise ratio. Only applicable to uniform or salt and pepper noises')
    parser.add_argument('-m', '--mean', type=float, default=0.0, help='Mean. Only applicable to gaussian noise')
    parser.add_argument('-s', '--stdev', type=float, default=0.01, help='Standard deviation. Only applicable to gaussian noise')

    # Set seed
    np.random.seed(0)

    # Parse args
    args = parser.parse_args()

    # Create output dir
    if not os.path.exists(args.output_dir):
        print("Creating {}".format(args.output_dir))
        os.makedirs(args.output_dir)

    # Prepare args
    noise_args = []
    if args.noise_type == "gaussian":
        noise_args.extend([args.mean, args.stdev])
    if args.noise_type in ("uniform", "salt_pepper"):
        noise_args.extend([args.ratio])

    # Add noise
    add_noise_to_images(args.input_dir, args.output_dir, _noise_functions[args.noise_type], noise_args)
