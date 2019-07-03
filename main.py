import os
import argparse
import configparser

from models import FSRCNN, IRCNN

import numpy as np
import tensorflow as tf

import logging
import logging.config

import preprocess

def get_model(config, input_shape=None):
    '''Creates a model from config
    :param config: config parser
    :return: Keras model
    '''
    # Get config
    model_name = config.get("default", "target_net")
    crop_size = config.getint("training", "patch_crop_size")
    upscale = config.getint("fsrcnn", "upscale")

    # Create model
    logging.getLogger().info("Creating model: {}".format(model_name))

    if model_name == "FSRCNN":

        if not input_shape:
            input_shape = (crop_size // upscale, crop_size // upscale, 1)
        model = FSRCNN(input_shape, upscale)
        loss = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(config.getfloat("training", "init_lr"))

    elif model_name == "IRCNN":

        if not input_shape:
            input_shape = (crop_size, crop_size, 3)
        model = IRCNN(input_shape)
        loss = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(config.getfloat("training", "init_lr"))

    else:
        raise ValueError("Not supported network {}".format(model_name))
        
    # Compile model
    model.compile(
        loss = loss,
        optimizer = optimizer,
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
    
    def epoch_begin(epoch, logs):
        print("Resetting seed")
        np.random.seed(0)

    callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_begin=epoch_begin))

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

    # Load training and validation datasets
    dataflow = preprocess.generate_training_data(data_path, config)
    val_dataflow = preprocess.generate_training_data(val_data_path, config)

    # Load steps
    steps = config.getint("training", "steps")
    val_steps = config.getint("training", "val_steps")
    
    # Train model
    model.fit_generator(dataflow, steps_per_epoch=steps, epochs=config.getint("training", "epochs"), verbose=1,
        callbacks=get_train_callbacks(config), validation_data=val_dataflow, validation_steps=val_steps, workers=0)

def infer(test_path, output_path, config):
    '''Train the network
    :param test_path: Test data directory path
    :param output_path: Output directory path
    :param config: config parser
    '''
    for name in os.listdir(test_path):
        # Load image
        image = preprocess.load_image(os.path.join(test_path, name))
        net_input = preprocess.preinference(image, config)

        # Create model
        model = get_model(config, net_input[0].shape)
            
        # Load model
        load_model(model, config)

        # Make prediction
        prediction = model.predict(net_input, verbose=1)[0]
        
        # Clip result
        prediction = preprocess.postinference(config, prediction, image)

        # Save images
        preprocess.save_image(name, prediction, output_path)

if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser(description='Train or infer CNN models')
    parser.add_argument('-t', '--train', action='store_true', help='Train the CNN')
    parser.add_argument('-d', '--data', type=str, default="train_data", help='Training data directory path')
    parser.add_argument('-v', '--val', type=str, default="val_data", help='Validation data directory path')
    parser.add_argument('-i', '--test', type=str, default="test_data", help='Test/Inference data directory path')
    parser.add_argument('-o', '--output', type=str, default="output", help='Test/Inference output directory path')
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
        logging.getLogger().info("Start training")
        train(args.data, args.val, config)
    else:
        logging.getLogger().info("Start inference")
        infer(args.test, args.output, config)

