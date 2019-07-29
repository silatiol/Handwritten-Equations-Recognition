import cv2
import pandas as pd  # Data processing, CSV file I/O
import numpy as np  # Linear Algebra
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


def Classify(model, input, intToLabels, numberProbabilities=3):

    predictions = model.predict(input)
    labelProbabilities = []
    for i in range(len(predictions)):
        print(np.sort(predictions[i]))
        p = np.argmax(predictions[i])
        labelProbabilities.append((intToLabels[p], predictions[i][p]))
    # a = np.array(labelProbabilities)

    dtype = [('label', str), ('probabilities', float)]
    # create a structured array
    a = np.array(labelProbabilities, dtype=dtype)

    # a = np.sort(a, order='probabilities')
    # a = a[::-1]
    # labelProbabilities.sort()
    return labelProbabilities


def ReadIndexChar(path):
    f = open(path, "r")
    lines = f.readlines()
    intToLabels = {}
    for l in lines:
        number = int(l.split(" ")[0])
        char = str(l.split(" ")[1]).strip()
        intToLabels[number] = char
    return intToLabels


def output(inputs, pred):
    out = np.zeros_like(image)
    print(inputs[0]['x'])
    for j in range(0, len(inputs)):
        cv2.putText(out, pred[j-1][0],
                    (inputs[j-1]['x'], inputs[j-1]['y']+inputs[j-1]['h']),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(inputs[j-1]['w'], inputs[j-1]['h'])/27.0,
                    (255, 255, 255))
    cv2.imshow("output", out)


image2 = cv2.imread("./img/2011/test/186.png")
image2 = cv2.bitwise_not(image2)
image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

cv2.imshow("test", image)

elements, hierachy = cv2.findContours(
    image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

kernel = np.ones((3, 3), np.uint8)

print(hierachy)
finals = []
inputs = []

i = 0
for e in elements:

    if(cv2.contourArea(e) < 10.0):
        continue

    x, y, w, h = cv2.boundingRect(e)
    inputs.append({'x': x, 'y': y, 'w': w, 'h': h})
    # k = cv2.drawContours(cv2.bitwise_not(
    #    np.zeros_like(image)), [e], 0, (0, 255, 0), 2)
    k = cv2.fillPoly(cv2.bitwise_not(
        np.zeros_like(image)), [e], color=(0, 255, 0))
    k = cv2.bitwise_not(k)
    # k = cv2.morphologyEx(k, cv2.MORPH_OPEN, kernel)
    # k = cv2.morphologyEx(k, cv2.MORPH_CLOSE, kernel)
    k = cv2.erode(k, np.ones((1, 1), np.uint8), iterations=1)

    if(w > h):
        while(w/h > 2):
            w = int(w*0.9)
        scale_percent = 28/w
    else:
        scale_percent = 28/h

    k = k[y:y+h, x:x+w]
    width = int(k.shape[1] * scale_percent)
    height = int(k.shape[0] * scale_percent)

    k = cv2.resize(k, (width, height))

    final = np.zeros((28, 28))
    y1, x1 = int((final.shape[0]-k.shape[0]) /
                 2), int((final.shape[1]-k.shape[1])/2)
    final[y1:y1+k.shape[0], x1:x1+k.shape[1]] = k
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    M = cv2.getRotationMatrix2D((28/2, 28/2), 270, 1)
    final = cv2.warpAffine(final, M, (28, 28))
    finals.append(np.divide(final.reshape(28, 28, 1), 255.0))
    # k = cv2.resize(k, (28, 28))
    # k = cv2.erode(k, kernel, iterations=1)
    cv2.imshow("test"+str(i), final)

    img2 = cv2.rectangle(image2, (x, y), (x+w, y+h), (0, 255, 0), 1)

    i += 1

# cv2.imshow("result", image2)
finals = np.array(finals)
print(finals[0])

model = tf.keras.models.load_model("model\CNN_19.model")
predictions = Classify(model, finals, ReadIndexChar("model\IndexChar.txt"))
print(predictions)
""" predictions = np.empty(6, dtype=np.unicode_)
predictions[0] = '2'
predictions[1] = '_'
predictions[2] = '2'
predictions[3] = '_'
predictions[4] = '3'
predictions[5] = '1' """

output(inputs, predictions)

while(1):
    q = cv2.waitKey(1) & 0xFF
    if q == 27:
        break
