from keras.models import Sequential 

# The various layers for the Neural Network Model.
# Dense - A layer that is fully connected (densely-connected.)
# Conv2D - A 2-dimensional convolutional layer.
# Dropout - A layer that helps prevent overfitting.
# Flatten - A layer that flattens the input.
# MaxPooling2D - A layer that performs Max Pooling of the Convolutions
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Since I have a GPU & I've GPU enabled, I am going to use the GPU version of keras 
# (NOTE: Ignore if you do not have GPU enabled)
import tensorflow as tf
from keras import backend as K
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.tensorflow_backend._get_available_gpus()

print ("START")

print("END")