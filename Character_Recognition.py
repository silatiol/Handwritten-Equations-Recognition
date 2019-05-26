import pandas as pd # Data processing, CSV file I/O 
import numpy as np # Linear Algebra
from sklearn.model_selection import train_test_split
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
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#K.tensorflow_backend._get_available_gpus()

print ("START")

IMAGE_SIZE = 28

#Read data and divided into train and set data

# columnsType = []
# selectedColumns = range(29)
# columnsType.append(tf.string)
# for i in range(IMAGE_SIZE):
#     columnsType.append(tf.int32)
# dataset = tf.data.experimental.CsvDataset(
#     "../data.csv", 
#     record_defaults = columnsType,
#     compression_type = None,
#     buffer_size = None,
#     header = True,
#     field_delim = ',',
#     use_quote_delim = True,
#     na_value = '',
#     select_cols=selectedColumns)

dataset = pd.read_csv(filepath_or_buffer="../data.csv", header=1)

trainSet, testSet = train_test_split(dataset,test_size = 0.01)


print("END")