"""
Image Analysis - Final Exam - Question 4 - Character ID'ing
from MNIST Dataset Using Neural Network, Custom Functions File
Created on Fri Dec  12 13:55:00 2025

@author: S.W. Marsden
"""

import numpy as np
import struct
#import os
from skimage import color
#from skimage.restoration import denoise_tv_chambolle
#from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import random

# setup the plot coordinates for plotting multiple images
def initialize_plot_locations(grid):
    # Create plot location coordinates. For a 3x3 grid of plots,
    # there are 9 locations each with a (row, col) coordinate.
    # PLOC[0] = (1,1); PLOC[1] = (1,2); PLOC[2] = (1,3)
    # PLOC[3] = (2,1); PLOC[4] = (2,2); PLOC[5] = (2,3)
    # PLOC[6] = (3,1); PLOC[7] = (3,2); PLOC[8] = (3,3)
    rows = grid[0]
    cols = grid[1]
    PLOC = np.zeros( ((rows * cols), 2) ).astype(np.uint8)
    idx = 0
    for row in range(0, rows):
        for col in range(0, cols):
            PLOC[idx] = (row+1, col+1)
            idx += 1
    idx = 0
    
    return PLOC;

# setup for plotting multiple images
def initialize_multiplot(plot_grid, text_size=8, res=800):
    plt.rc('font', size=text_size)
    fig, ax = plt.subplots(plot_grid[0], plot_grid[1],
                           dpi=res, constrained_layout=True)
    return ax;

# plot an array of data or an image on a multiplot layout
def ezplot(plotInfo, plotObj, plotType='image', colors='gray', gridshare=0):
    # plotInfo is a 3 wide array with data, ploc, and name.
    rowI = plotInfo[1][0].astype(int) - 1 # plotInfo[1] is ploc
    colI = plotInfo[1][1].astype(int) - 1 
    data = plotInfo[0]
    name = plotInfo[2]
    ax = plotObj
    
    if plotType == 'image':
        ax[rowI][colI].imshow(data, cmap=colors)
        ax[rowI][colI].set_title(name)
        ax[rowI][colI].set_xticks([])
        ax[rowI][colI].set_yticks([])
        
    if plotType == 'plot':
        ax[rowI][colI].plot(data[0], data[1])
        ax[rowI][colI].set_title(name)
    
    if plotType == 'pointOverlay':
        ax[rowI][colI].autoscale(False)
        ax[rowI][colI].plot(data[:, 1], data[:, 0], colors,
                            markeredgewidth=0.25, markersize=3,
                            markerfacecolor='w')
    
    if plotType == 'contour':
        ax[rowI][colI].plot(data[:, 1], data[:, 0], '-b', lw=3)
    
    if gridshare == 1:
        ax[rowI][colI].sharex(ax[0][0])
        ax[rowI][colI].sharey(ax[0][0])
   
    return 0;

# grayscale 2, 3, or 4 channel images
def convert_to_gray(img):
    img_dimension = len(img.shape)
    if img_dimension == 2:
        return img;
    elif img_dimension == 3:
        img = color.rgb2gray(img[:,:,0:3])
        """minimum_val = np.min(img)
        maximum_val = np.max(img)
        img = (img - minimum_val) / (maximum_val - minimum_val)
        img = np.array((255 * img))
        """
        # if there are more than 3 color params (RGBA or CMYK),
        # only 0 thru 3 are used.
        return img;
    else:
        print("ERROR while grayscaling.")
        return 1;

# take a square section from top left of an image for detail analysis
def snip(img, sizeX=300, sizeY=300):
    img_snip = img[:sizeY,:sizeX]
    return img_snip;

# read in an idx1-ubyte file and create a N-large list of numbers 0 to 9.
def load_idx1_ubyte(folder, filename):
    file = open(f"{folder}/{filename}", "rb")
    # Read the magic number and number of items (both are 4-byte integers, Big-Endian)
    # '>' indicates big-endian, 'II' indicates 2 unsigned integers
    magic, size = struct.unpack('>II', file.read(8))
    # Verify the magic number (should be 2049 for label files)
    if magic != 2049:
        raise ValueError(f'Invalid magic number: {magic}. Expected 2049 for IDX1 label file.')
    # Read the label data as a 1D numpy array of unsigned bytes (uint8)
    # '>' ensures the correct byte order
    labels = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
    # Verify the number of labels read matches the size specified in the header
    if labels.shape[0] != size:
        raise ValueError(f'Data size mismatch. Expected {size}, but read {labels.shape[0]}.')
    
    file.close()
    return labels;

# read in an idx1-ubyte file and create an array of images from the data
def load_idx3_ubyte(folder, filename):
    file = open(f"{folder}/{filename}", "rb")

    # ~~~~~~~~~~~~~~~~~~~~~~~~ idx3-ubyte format ~~~~~~~~~~~~~~~~~~~~~~~~
    #
    #                         datatype(1000 = ubyte) v
    # magic num   = [0000 0000]   [0000 0000]   [0000 1000]   [0000 0011]
    #                    dataDimensions(11 = 3: count, rows, cols) ^
    #
    # num of imgs = [0000 0000]   [0000 0000]   [0010 0111]   [0001 0000]
    #                                  10k imgs |--->               <---|
    #
    # num of rows = [0000 0000]   [0000 0000]   [0000 0000]   [0001 1100]
    #                                                 23 rows |---> <---|
    #
    # num of cols = [0000 0000]   [0000 0000]   [0000 0000]   [0001 1100]
    #                                                 23 cols |---> <---|
    #
    #               pixel #1      pixel #2      pixel #3
    # img_data =    [0000 0000]   [0111 1111]   [1111 1111] ...
    #               = 0           = 128         = 255
    
    # Read the magic number, number of imgs, and dimension sizes.
    # '>' indicates big-endian, 'IIII' indicates 4 unsigned integers
    magic, img_count, rows, cols = struct.unpack('>IIII', file.read(16))

    if magic != 2051: # 2051 is the magic number for idx3-ubyte files
            raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')

    # Read the label data as a 1D numpy array of unsigned bytes (uint8)
    # '>' ensures the correct byte order
    img_data = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
    
    file.close()
    
    # reshape img_data into list of 2D-arrays.
    imgs = img_data.reshape(img_count, rows, cols)
    
    return imgs;

# confusion matrix
def find_confusion(predictions, answers, digits=10):
    # for decimal digits 0 thru 9:
    #          ACTUAL  VALUE
    #       0 1 2 3 4 5 6 7 8 9
    # P  0  $ x x x x x x x x x   e.g. given an actual value of 8,
    # R  1  x $ x x x x x x x x        if the model predicts a 2 then
    # E  2  x x $ x x x x x x x        row 2 col 8 increments by 1,
    # D  3  x x x $ x x x x x x        indicating a false negative for 8
    # I  4  x x x x $ x x x x x        as well as a false positive for 2.
    # C  5  x x x x x $ x x x x        But if the model correctly predicts
    # T  6  x x x x x x $ x x x        an 8, then row 8 col 8 increments,
    # I  7  x x x x x x x $ x x        indicating a true positive for 8.
    # O  8  x x x x x x x x $ x
    # N  9  x x x x x x x x x $
    #
    # metrics:           truePositives($)
    #          ------------------------------------ = precision (for 1 digit)
    #           TPs($) + falsePositives(x's in row)
    #
    #                         TPs($)
    #          ------------------------------------ = recall (for 1 digit)
    #           TPs($) + falseNegatives(x's in col)
    #
    # high accur.: fewer false pos/negs (but more true positives missed)
    # high recall: fewer true positives missed (but more false positives)
    conMat = np.zeros((digits, digits))
    for n in range(len(answers)):
        digit = answers[n]
        guess = predictions[n]
        conMat[guess, digit] += 1 # if guess == digit -> true(+) at that num
                                  # otherwise -> false(+)/(-) at that num
    precision = np.zeros(digits)
    recall = np.zeros(digits)
    for d in range(0, digits):
        FNs = np.sum(conMat[:, d]) # sum all values of col d (current digit)
        TPs = conMat[d,d]          # get true(+)'s ($'s on diagonal)
        FNs -= TPs                 # subtract true(+)'s to get false(-)'s
        FPs = np.sum(conMat[d, :]) # sum all values of row d (current digit)
        FPs -= TPs                 # subtract true(+)'s to get false(+)'s
        precision[d] = TPs / (TPs + FPs)
        recall[d] = TPs / (TPs + FNs)
    
    return conMat, precision, recall;

# create a list of ints, taking 1 index from the dataset no more than once
def generate_data_order(max_val, samples):
    dataset = np.arange(0, max_val, 1).tolist()
    data_order = random.sample(dataset, samples)
    
    return data_order;

# take generated weights and a testing set and evaluate performance
def test_net(test_data, test_answers, order, weights):
    errors = []
    predictions = []
    answers = []
    
    mn = np.min(test_data)
    mx = np.max(test_data)
    test_data = (test_data - mn) / (mx - mn)
    
    for n in range(len(order)): # iterate through the test set
        i = order[n]
        data_vector = test_data[i].reshape(784, 1)
        x = data_vector + 1
        x_count = len(x)
        y = 0
        for k in range(0, x_count): # iterate through the inputs
            y += (weights[k] * x[k])
        predictions.append(sigmoid(y)*10)
        errors.append(error(predictions[n], test_answers[i]))
        answers.append(test_answers[i])
        if n==100:
            print(f"y={y}, sigmoid(y)={sigmoid(y)*10}, error={error(predictions[n], test_answers[i])}")
    
    predictions = np.concat(predictions).tolist()
    errors = np.concat(errors).tolist()
    answers = np.array(answers).tolist()
    
    return predictions, answers, errors;

# neural network
def build_net(inputs, target, iters, init_weights=[0], learn_rate=0.05):
    # take in inputs and establish inital weights (if needed)
    x = inputs + 1 # nx1 col vector, e.g. 25x25 pixels -> [x0:x783].
                   # +1 is added here to ensure x[n] != 0, which messes up y.
    x_count = len(x)
    if len(init_weights) < x_count:
        w = np.random.rand(x_count, 1) # random values from [0, 1) in col vector.
    else:
        w = init_weights
    
    # begin iterating through the fwd-pass/bck-pass cycle
    prediction = []
    net_error = []
    found = 0
    for k in range(iters):
        # Forward Pass
        y = 0
        for n in range(0,x_count): # sum of products
            y += w[n]*x[n]
        y = y/50 # ensures that the sum of w*x is around 9 to
                 # avoid saturating the sigmoid function.
        predicted = sigmoid(y)*10 # ensures that the prediction is properly
                                  # scaled for the digits 0 -> 9
                                  # (predicts floats 0 -> 1 otherwise)
        err = error(predicted, target)
        if err < 8e-5 and found == 0:
            found = k
        prediction.append(predicted)
        net_error.append(err)
        
        # Backward Pass
        g1 = error_predicted_deriv(predicted, target)
        g2 = sigmoid_sop_deriv(y)
        g3w = []
        for n in range(0,x_count):
            g3w.append(sop_w_deriv(x[n]))
        gradw = []
        for n in range(0,x_count):
            gradw.append(g3w[0] * g2 * g1)
        
        for n in range(0,x_count):
            w[n] = update_w(w[n], gradw[n], learn_rate)
        
    prediction = np.concat(prediction).tolist()
    net_error = np.concat(net_error).tolist()
    w = np.concat(w).tolist()
    
    return prediction, net_error, w, found;

def sigmoid(sop):
    return 1.0/(1+np.exp(-1*sop))

def error(predicted, target):
    return np.power(predicted-target, 2)

def error_predicted_deriv(predicted, target):
    return 2*(predicted-target)

def sigmoid_sop_deriv(sop):
    return sigmoid(sop)*(1.0-sigmoid(sop))

def sop_w_deriv(x):
    return x

def update_w(w, grad, learning_rate):
    return w - learning_rate*grad
    
    
    
    
    
    
    
    
    
    