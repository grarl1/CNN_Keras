import argparse
import configparser

from models import IRCNN

import tensorflow as tf

import logging
import logging.config

import preprocess

def train(data_path, val_data_path, config):
    '''Train the network
    :param data_path: Training data directory path
    :param val_data_path: Validation data directory path
    :param config: config parser
    '''

    # Get model
    model = IRCNN()
    print(model.summary())

    # Compile model
    model.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = tf.train.AdamOptimizer(0.001),
    )

    # Load training and validation datasets
    (data, val_data) = preprocess.load_training_data(data_path, val_data_path, config)

    # Preprocess data
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function = preprocess.add_random_noise,
        data_format = "channels_last"
    )
    dataflow = datagen.flow(data, data, batch_size=config.getint("training", "batch_size"), save_to_dir="flow")

    ## Train model
    model.fit_generator(dataflow, epochs=1, steps_per_epoch=10,  verbose=1) # TODO bath=256

def infer():
    pass

if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser(description='Train or infer using IRCNN model')
    parser.add_argument('-t', '--train', action='store_true', help='Train the CNN')
    parser.add_argument('-d', '--data', type=str, default="train_data", help='Training data directory path')
    parser.add_argument('-v', '--val', type=str, default="val_data", help='Validation data directory path')
    parser.add_argument('-i', '--test', type=str, default="test_data", help='Test/Inference data directory path')
    parser.add_argument('-c', '--config', type=str, default="config.ini", help='Network configuration file')

    # Parse arguments
    args = parser.parse_args()

    # Setup logger
    logging.config.fileConfig('logging.ini')

    # Read configuration
    config = configparser.ConfigParser()
    config.read(args.config)

    # Process arguments
    if args.train:
        train(args.data, args.val, config)
    else:
        infer()

