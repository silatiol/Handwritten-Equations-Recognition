import cv2 
import context as con
import pandas as pd  # Data processing, CSV file I/O
import numpy as np  # Linear Algebra
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageChops
from skimage import img_as_ubyte
from skimage.io import imread
from skimage.filters import gaussian, threshold_minimum
from skimage.morphology import square, erosion, thin

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


def binarize(image_abs_path):

    # Convert color image (3-channel deep) into grayscale (1-channel deep)
    # We reduce image dimensionality in order to remove unrelevant features like color.
    grayscale_img = imread(image_abs_path, as_grey=True)

    # Apply Gaussian Blur effect - this removes image noise
    gaussian_blur = gaussian(grayscale_img, sigma=1)

    # Apply minimum threshold
    thresh_sauvola = threshold_minimum(gaussian_blur)

    # Convert thresh_sauvola array values to either 1 or 0 (white or black)
    binary_img = gaussian_blur > thresh_sauvola

    return binary_img


def shift(contour):

    # Get minimal X and Y coordinates
    x_min, y_min = contour.min(axis=0)[0]

    # Subtract (x_min, y_min) from every contour point
    return np.subtract(contour, [x_min, y_min])


def get_scale(cont_width, cont_height, box_size):

    ratio = cont_width / cont_height

    if ratio < 1.0:
        return box_size / cont_height
    else:
        return box_size / cont_width


def extract_patterns(image_abs_path):

    max_intensity = 1
    # Here we define the size of the square box that will contain a single pattern
    box_size = 28

    binary_img = binarize(image_abs_path)

    # Apply erosion step - make patterns thicker
    eroded_img = erosion(binary_img, selem=square(3))

    # Inverse colors: black --> white | white --> black
    binary_inv_img = max_intensity - eroded_img

    # Apply thinning algorithm
    thinned_img = thin(binary_inv_img)

    # Before we apply opencv method, we need to convert scikit image to opencv image
    thinned_img_cv = img_as_ubyte(thinned_img)

    # Find contours
    contours, _ = cv2.findContours(
        thinned_img_cv, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right (sort by bounding rectangle's X coordinate)
    contours = sorted(contours, key=lambda cont: cv2.boundingRect(cont)[0])

    # Initialize patterns array
    patterns = []
    i = 0
    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        expression.append({'x': x, 'y': y, 'w': w, 'h': h})

        # Initialize blank white box that will contain a single pattern
        pattern = np.zeros(shape=(box_size, box_size), dtype=np.uint8) * 255

        # Shift contour coordinates so that they are now relative to its square image
        shifted_cont = shift(contour)

        # Get size of the contour
        cont_width, cont_height = cv2.boundingRect(contour)[2:]
        # boundingRect method returns width and height values that are too big by 1 pixel
        # cont_width -= 1
        # cont_height -= 1

        # Get scale - we will use this scale to interpolate contour so that it fits into
        # box_size X box_size square box.
        scale = get_scale(cont_width, cont_height, box_size)

        # Interpolate contour and round coordinate values to int type
        rescaled_cont = np.floor(shifted_cont * scale).astype(dtype=np.int32)

        # Get size of the rescaled contour
        rescaled_cont_width, rescaled_cont_height = cont_width * scale, cont_height * scale

        # Get margin
        margin_x = int((box_size - rescaled_cont_width) / 2)
        margin_y = int((box_size - rescaled_cont_height) / 2)

        # Center pattern wihin a square box - we move pattern right by a proper margin
        centered_cont = np.add(rescaled_cont, [margin_x, margin_y])

        # Draw centered contour on a blank square box
        cv2.drawContours(pattern, [centered_cont],
                         contourIdx=0, color=(255, 255, 255))

        # Invert row and cols (because tensorflow read them opposite way)
        pattern = cv2.flip(pattern, 1)

        M = cv2.getRotationMatrix2D((box_size/2, box_size/2), 90, 1)
        pattern = cv2.warpAffine(pattern, M, (box_size, box_size))

        patterns.append(
            np.divide(pattern.reshape(box_size, box_size, 1), 255.0))
        cv2.imshow("tert"+str(i),
                   np.divide(pattern.reshape(box_size, box_size, 1), 255.0))
        i += 1

    return patterns


def widthFirst(elem):
    return elem['w']


def latex_to_img(tex):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.axis('off')
    plt.text(0.05, 0.5, f'${tex}$', fontsize=32)
    plt.show()


def Classify(model, input, intToLabels, numberProbabilities=3):

    predictions = model.predict(input)
    labelProbabilities = []
    for i in range(len(predictions)):
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


def getExpression(inp, pred):
    topCut = 0
    bottomCut = 0
    minY = image.shape[0]
    maxY = 0
    for j in range(len(pred)):
        inp[j]['character'] = pred[j][0]
        inp[j]['center'] = (inp[j]['x']+(inp[j]['w']/2),
                            inp[j]['y']+(inp[j]['h']/2))
        if(inp[j]['y']+inp[j]['h'] > maxY):
            maxY = inp[j]['y']+inp[j]['h']
        if(inp[j]['y'] < minY):
            maxY = inp[j]['y']
    topCut = (maxY-minY) * 0.35
    bottomCut = (maxY-minY) * 0.65
    return inp, topCut, bottomCut


def contextAnalisys(exp):
    result = []
    for j in range(len(exp)):
        if(j >= len(exp)):
            break
        if('active' not in exp[j].keys()):
            exp[j]['active'] = False
            if (exp[j]['character'] == '-'):
                over = con.findOver(exp, j)
                under = con.findUnder(exp, j)
                if(len(over) == 0 and len(under) == 0):
                    result.append(exp[j])
                elif(len(over) == 1 and len(under) == 0):
                    # symb like = or +- or += or >= etc
                    over[0]['active'] = False
                    if(over[0]['character'] == '-'):
                        result.append(
                            {'character': '=', 'x': exp[j]['x'], 'y': exp[j]['y'],
                             'w': exp[j]['w'], 'h': exp[j]['h'], 'center': exp[j]['center']})
                    elif(over[0]['character'] == '+'):
                        result.append(
                            {'character': '\pm', 'x': exp[j]['x'], 'y': exp[j]['y'],
                             'w': exp[j]['w'], 'h': exp[j]['h'], 'center': exp[j]['center']})
                    elif(over[0]['character'] == '>'):
                        result.append(
                            {'character': '\geq', 'x': exp[j]['x'], 'y': exp[j]['y'],
                             'w': exp[j]['w'], 'h': exp[j]['h'], 'center': exp[j]['center']})
                    elif(over[0]['character'] == '<'):
                        result.append(
                            {'character': '\leq', 'x': exp[j]['x'], 'y': exp[j]['y'],
                             'w': exp[j]['w'], 'h': exp[j]['h'], 'center': exp[j]['center']})
                elif(len(over) == 0 and len(under) == 1):
                    under[0]['active'] = False
                    if(under[0]['character'] == '-'):
                        result.append(
                            {'character': '=', 'x': exp[j]['x'], 'y': exp[j]['y'],
                             'w': exp[j]['w'], 'h': exp[j]['h'], 'center': exp[j]['center']})
                    else:
                        result.append(
                            {'character': '\pmod', 'bottom': contextAnalisys(under), 'x': exp[j]['x'], 'y': exp[j]['y'], 'w': exp[j]['w'], 'h': under[0]['h'], 'center': exp[j]['center']})
                elif(len(over) == 0 and len(under) > 1):
                    result.append(
                        {'character': '\pmod', 'bottom': contextAnalisys(under), 'x': exp[j]['x'], 'y': exp[j]['y'], 'w': exp[j]['w'], 'h': exp[j]['h'], 'center': exp[j]['center']})
                else:
                    result.append({'character': 'frac', 'bottom': contextAnalisys(
                        under), 'over': contextAnalisys(over), 'x': exp[j]['x'], 'y': (over[0]['y']), 'w': exp[j]['w'], 'h': under[0]['y']+under[0]['h']-(over[0]['y']), 'center': exp[j]['center']})

            elif (exp[j]['character'] == 'sqrt'):
                inner = con.findInner(exp, j)
                result.append(
                    {'character': 'sqrt', 'bottom': contextAnalisys(inner), 'x': exp[j]['x'], 'y': exp[j]['y'], 'w': exp[j]['w'], 'h': exp[j]['h'], 'center': exp[j]['center']})
            else:
                result.append(exp[j])

    return result


def output(inputs, pred):
    out = np.zeros_like(image)
    for j in range(0, len(inputs)):
        cv2.putText(out, pred[j-1][0],
                    (inputs[j-1]['x'], inputs[j-1]['y']+inputs[j-1]['h']),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(inputs[j-1]['w'], inputs[j-1]['h'])/27.0,
                    (255, 255, 255))
    cv2.imshow("output", out)


path = "./img/2011/test/319.png"
image2 = cv2.imread(path)
image2 = cv2.bitwise_not(image2)
image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
cv2.imshow("input", image)

finals = []
expression = []

finals = np.array(extract_patterns(path))

model = tf.keras.models.load_model("C:\\Users\\snake\\Downloads\\CNN_19.model")
predictions = Classify(model, finals, ReadIndexChar("model\IndexChar.txt"))

expression, topCut, bottomCut = getExpression(expression, predictions)

sorted(expression, key=widthFirst)

# print(expression)
expression = contextAnalisys(expression)


latex = con.toLatex(expression)

print(latex)
latex_to_img(latex)
# output(expression, predictions)

""" while(1):
    q = cv2.waitKey(1) & 0xFF
    if q == 27:
        break
 """
