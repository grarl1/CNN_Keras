# CNN Keras
Implementation of some CNN models using [Keras: The Python Deep Learning library](https://keras.io/)

## Implemented models

### FSRCNN
Implementation of FSRCNN model based on [Accelerating the Super-Resolution Convolutional Neural Network](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)

### IRCNN
Implementation of IRCNN model Keras based on [Learning Deep CNN Denoiser Prior for Image Restoration](http://www4.comp.polyu.edu.hk/~cslzhang/paper/IRCNN_CVPR17.pdf)

## Saved checkpoints
This repository contains some pre-trained models that can be enabled by creating symbolic links to the checkpoint directory and the configuration file in the root path. This way the code will detect these files and will load the weights of the corresponding model.

### FSRCNN
```
ln -s saved_checkpoints/FSRCNN/checkpoint
ln -s saved_checkpoints/FSRCNN/config.ini
```

### IRCNN
```
ln -s saved_checkpoints/IRCNN/checkpoint
ln -s saved_checkpoints/IRCNN/config.ini
```

## Usage
### For training
`python main.py --train --data /path/to/<training images dir> --val /path/to/<validation images dir> -c config.ini`

### For inference
`python main.py --test /path/to/<test images dir> --output /path/to/<output dir> -c config.ini`
