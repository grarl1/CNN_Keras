import os
import argparse
import configparser

from models import IRCNN

import tensorflow as tf

import logging
import logging.config

import preprocess

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
        preprocessing_function = preprocess.add_random_noise,
        data_format = "channels_last"
    )
    dataflow = datagen.flow(x=data, y=data, batch_size=batch_size)

    # Preprocess validation data
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function = preprocess.add_random_noise,
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
    model = IRCNN()
    print(model.summary())
    
    # Load model
    checkpoint_path = config.get("default", "checkpoint_path")
    if os.path.isfile(checkpoint_path):
        logging.getLogger().info("Checkpoint found. Loading weights.")
        model.load_weights(checkpoint_path)

    # Compile model
    model.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(config.getfloat("training", "init_lr")),
    )

    # Load training and validation datasets
    (data, val_data) = preprocess.load_training_data(data_path, val_data_path, config)
    (dataflow, val_dataflow) = get_generators(data, val_data, config)
    
    # Train model
    batch_size = config.getint("training", "batch_size")
    steps_per_epoch = len(data) // batch_size
    validation_steps = len(val_data) // batch_size

    model.fit_generator(dataflow, steps_per_epoch=steps_per_epoch, epochs=config.getint("training", "epochs"), verbose=1,
        callbacks=get_train_callbacks(config), validation_data=val_dataflow, validation_steps=validation_steps)

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

