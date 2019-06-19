import os
import argparse
import configparser

from models import IRCNN

import numpy as np
import tensorflow as tf

import logging
import logging.config

import preprocess
import noise

def get_model(config, input_shape=None):
    '''Creates a model from config
    :param config: config parser
    :return: Keras model
    '''
    # Create model
    if not input_shape:
        crop_size = config.getint("training", "patch_crop_size")
        input_shape = (crop_size, crop_size, 3)
    model = IRCNN(input_shape)

    # Compile model
    model.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(config.getfloat("training", "init_lr")),
    )

    return model

def load_model(model, config):
    '''Loads model from path taken from config
    :param model: Keras model
    :param config: config parser
    '''
    checkpoint_dir = config.get("default", "checkpoint_dir")
    checkpoint_path = config.get("default", "checkpoint_path")
    if os.path.isdir(checkpoint_dir):
        logging.getLogger().info("Checkpoint found. Loading weights.")
        model.load_weights(checkpoint_path)

def save_model(model, config):
    '''Save model to path taken from config
    :param model: Keras model
    :param config: config parser
    '''
    checkpoint_dir = config.get("default", "checkpoint_dir")
    checkpoint_path = config.get("default", "checkpoint_path")
    logging.getLogger().info("Saving model weights")
    model.save_weights(checkpoint_path)

def get_generators(data, val_data, config):
    '''Return generators for training data and validation data
    :param data: numpy array containing the list of training images
    :param val_data: numpy array containing the list of validation images
    :param config: config parser
    '''
    # Get batch size
    batch_size = config.getint("training", "batch_size")

    # Preprocess data
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function = noise.add_random_noise,
        data_format = "channels_last"
    )
    dataflow = datagen.flow(x=data, y=data, batch_size=batch_size)

    # Preprocess validation data
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function = noise.add_random_noise,
        data_format = "channels_last"
    )
    val_dataflow = val_datagen.flow(x=val_data, y=val_data, batch_size=batch_size)

    return (dataflow, val_dataflow)

def get_train_callbacks(config):
    '''Returns a list of keras callbacks
    :param config: config parser
    :return: list of keras callbacks
    :rtype: list(tf.keras.Callback)
    '''
    callbacks = []

    callbacks.append(tf.keras.callbacks.CSVLogger(
        config.get("default", "csv_log"),
        separator=',',
        append=True
    ))

    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=config.getint("training", "reduce_lr_patience"),
        verbose=1,
        mode='min',
        min_lr=config.getfloat("training", "min_lr"),
    ))

    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=config.getint("training", "early_stopping_patience"),
        verbose=1,
        mode='min',
        baseline=None,
        restore_best_weights=True
    ))

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        config.get("default", "checkpoint_path"),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        period=config.getint("training", "checkpoint_period"),
    ))

    return callbacks

def train(data_path, val_data_path, config):
    '''Train the network
    :param data_path: Training data directory path
    :param val_data_path: Validation data directory path
    :param config: config parser
    '''
    # Create model
    model = get_model(config)
    logging.getLogger().info(model.summary())
        
    # Load model
    load_model(model, config)

    # Save model info
    with open(config.get("default", "model_json"), "w") as json_file:
        json_file.write(model.to_json())

    # Load training and validation datasets
    (data, val_data) = preprocess.load_training_data(data_path, val_data_path, config)
    (dataflow, val_dataflow) = get_generators(data, val_data, config)
    
    # Train model
    batch_size = config.getint("training", "batch_size")
    steps_per_epoch = len(data) // batch_size
    validation_steps = len(val_data) // batch_size

    model.fit_generator(dataflow, steps_per_epoch=steps_per_epoch, epochs=config.getint("training", "epochs"), verbose=1,
        callbacks=get_train_callbacks(config), validation_data=val_dataflow, validation_steps=validation_steps)

def infer(test_path, output_path, config):
    '''Train the network
    :param test_path: Test data directory path
    :param output_path: Output directory path
    :param config: config parser
    '''
    # Load test images
    test_images = preprocess.load_images(test_path)
    test_images = preprocess.normalize_images(test_images)

    for name, image in zip(os.listdir(test_path), test_images):
        # Create model
        model = get_model(config, image.shape)
            
        # Load model
        load_model(model, config)

        # Make prediction
        image = image.reshape((1, *image.shape))
        prediction = model.predict(image, verbose=1)
        
        # Clip
        prediction = np.clip(prediction, 0, 1)[0]

        # Save images
        preprocess.save_image(name, prediction, output_path)

if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser(description='Train or infer using IRCNN model')
    parser.add_argument('-t', '--train', action='store_true', help='Train the CNN')
    parser.add_argument('-d', '--data', type=str, default="train_data", help='Training data directory path')
    parser.add_argument('-v', '--val', type=str, default="val_data", help='Validation data directory path')
    parser.add_argument('-i', '--test', type=str, default="test_data", help='Test/Inference data directory path')
    parser.add_argument('-o', '--output', type=str, default="output", help='Test/Inference output directory path')
    parser.add_argument('-c', '--config', type=str, default="config.ini", help='Network configuration file')

    # Set random seed
    np.random.seed(0)

    # Parse arguments
    args = parser.parse_args()

    # Setup logger
    logging.config.fileConfig('logging.ini')

    # Read configuration
    config = configparser.ConfigParser()
    config.read(args.config)

    # Process arguments
    if args.train:
        logging.getLogger().info("Start training")
        train(args.data, args.val, config)
    else:
        logging.getLogger().info("Start inference")
        infer(args.test, args.output, config)

