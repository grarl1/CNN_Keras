import tensorflow as tf
from tensorflow.keras import layers

def IRCNN(input_shape):

    # Attributes
    seed = 0
    kernel_weight_decay = 1e-4
    dilation_rate = [1, 2, 3, 4, 3, 2, 1]

    # Create model
    model = tf.keras.models.Sequential()

    # First layer
    model.add(layers.Conv2D(
            input_shape=input_shape,
            filters=64,
            kernel_size=[3, 3], 
            strides = 1,
            padding = 'same',
            data_format = "channels_last",
            dilation_rate = dilation_rate[0],
            activation = tf.nn.relu,
            kernel_initializer = tf.keras.initializers.glorot_uniform(seed=seed),
            kernel_regularizer = tf.keras.regularizers.l2(kernel_weight_decay)
    ))

    # Middle layers
    for i in range(2, 7):
        model.add(layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            strides = 1,
            padding = 'same',
            data_format = "channels_last",
            dilation_rate = dilation_rate[i-1],
            activation = None,
            kernel_initializer = tf.keras.initializers.glorot_uniform(seed=seed),
            kernel_regularizer = tf.keras.regularizers.l2(kernel_weight_decay)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(tf.nn.relu))


    # Last layer
    model.add(layers.Conv2D(
            filters=3,
            kernel_size=[3, 3],
            strides = 1,
            padding = 'same',
            data_format = "channels_last",
            dilation_rate = dilation_rate[-1],
            activation = None,
            kernel_initializer = tf.keras.initializers.glorot_uniform(seed=seed),
            kernel_regularizer = tf.keras.regularizers.l2(kernel_weight_decay)
    ))

    return model
