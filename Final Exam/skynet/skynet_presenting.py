"""
Image Analysis - Final Exam - Question 4 - Character ID'ing
from MNIST Dataset Using Neural Network, Custom Neural Network File
Created on Sun Dec  14 11:11:00 2025

@author: S.W. Marsden
"""

import random
import numpy as np
import numpy.random as rand
#import struct
#import os
#from skimage import color
#from skimage.restoration import denoise_tv_chambolle
#from scipy.ndimage import convolve
#import matplotlib.pyplot as plt

# used for normalization of input values in following classes and functions.
norm_factor = 0.1

class Neuron:
    def __init__(self, inputs, weights, scale=1):
        self.x = inputs # no bias implementation yet.
        self.w = weights
        self.scale = scale
        self.s = sop(inputs, weights)
        self.y = scale * sigmoid(self.s)
        
    def refresh(self): # recalculate sum-of-products (s) and output (y).
        self.s = sop(self.x, self.w)
        self.y = self.scale * sigmoid(self.s)
    
    def set_weights(self, weights):
        self.w = weights
        self.refresh()
    
    def set_inputs(self, inputs):
        self.x = inputs
        self.x = np.reshape(self.x, (len(self.x),1))
        self.refresh()
        
    def set_scale(self, scale):
        self.scale = scale
        self.refresh()

class NeuralNet:
    def __init__(self, neurons, inputs, weights, scale=1):
        self.neurons = neurons # 1D array of Neuron objects.
        self.x = inputs
        self.w = weights
        self.scale = scale
        self.y = self.neurons[-1].y # output of the last node (output node).
        self.xnum = len(self.x)
        self.wnum = len(self.w)
        self.nodenum = len(self.neurons)
        
    def refresh(self):
        # run when weights are set, inputs are set, and/or scale is set.
        # DOES NOT RUN AUTOMATICALLY, run manually to prevent redundant refreshing.
        # recalc output from inner layer, pass to output layer,
        # and recalc the output of the output node.
        inner_layer_outputs = []
        curr_weights = []
        for n in range(0, self.nodenum - 1): # retrieve the outputs of the inner layer nodes.
            self.neurons[n].refresh()
            inner_layer_outputs.append(self.neurons[n].y)
            curr_weights.append(self.neurons[n].w)
        self.neurons[-1].set_inputs(inner_layer_outputs) # set these as inputs to the output node.
        self.y = self.neurons[-1].y # update the output of neural network.
        curr_weights.append(self.neurons[-1].w)
        self.w = curr_weights
        
    def set_inner_weights(self, weights): # change weights of the hidden layer
        if len(weights) != self.xnum * (self.nodenum - 1):
                print("ERROR while setting inner weights, num. of weights does not fit this num. of inputs.")
                return 1;
        w_start = 0
        for n in range(0, self.nodenum - 1): # iterate through all but the last node.
            w_stop = self.xnum + w_start
            self.neurons[n].set_weights(weights[w_start:w_stop])
            w_start = w_stop
        self.w = weights[0:w_stop]
    
    def set_output_weights(self, weights): # change weights of the hidden layer
        if len(weights) != self.nodenum - 1:
            print("ERROR while setting output weights, num. of weights does not fit this num. of inputs.")
            return 1;
        self.neurons[-1].set_weights(weights)
        self.w = weights[self.wnum - self.nodenum + 1 : self.wnum]
        
    def set_inputs(self, inputs): # change inputs of the hidden layer
        if len(inputs) != self.xnum:
            print("ERROR while setting inputs, num. of inputs does not match this network.")
            return 1;
        if len(inputs) > self.nodenum:
            self.x = normalize(inputs)
        else:
            self.x = inputs
        for n in range(0, self.nodenum - 1):
            self.neurons[n].set_inputs(self.x)
    
    def apply_dropout(self, probability=0.5): # set 50% of hidden layers outputs to 0.
        drop_num = round(probability*(self.nodenum-1))
        node_list = range(0, self.nodenum-1)
        drop_nodes = random.sample(node_list, drop_num)
        #print(drop_nodes)
        for n in range(0, drop_num):
            self.neurons[drop_nodes[n]].y = 0
        self.neurons[-1].refresh()
        self.y = self.neurons[-1].y
        
        
# construct a neural network in the following format:
# input0 ⭶⭷ neuron0 ⭶
# input1 ⭰⭲ neuron1 ⭰⭲ neuronN+1 ⭰⭲ output
# inputN ⭹⭸ neuronN ⭹
# all inputs connected to all neurons in inner layer, and
# all neurons in inner layer connected to the output neuron.
def create_neuralNet(inputs, weights=[0], neurons=10, scale=9.0):
    # inputs is an N x 1 vector of data being interpreted.
    # weights is a M x 1 vector of scalars, 1 for each input -> node connection.
    # --->> given 100 inputs and 10 neurons...
    # --->> there are 100 connections per 1 neuron.
    # --->> and given 9 neurons in inner layer and +1 in output layer...
    # --->> yields 900 connections in inner layer and +9 in output layer
    # --->> therefore: M = 909 weights.
    # --->> wnum = xnum * (neurons - 1) + (neurons - 1)
    # --->>      = (neurons - 1) * (1 + xnum)
    # neurons is the desired size of the network, with 1 neuron for the
    # output layer and the rest for the inner layer.
    # scale determines what number the output layer should be producing.
    # normally 0.00 -> 0.999..., so for digits 0.00 to 9.00 scale = 9.
    
    # take in the inputs and weights.
    """x = normalize(inputs, scale=norm_factor*(10/neurons)) # scale here helps downsize the
                                             # large number of inputs to prevent
                                             # saturating the sigmoid(s) calc."""
    x = inputs
    
    xnum = len(x)
    wnum = (1 + xnum) * (neurons - 1)
    if len(weights) == 1: # if true, weights have not been provided.
        w = generate_random_weights((neurons - 1) * (1 + xnum))
    elif len(weights) != wnum: # check for incorrect number of weights.
        print("ERROR while creating neuralNet, num. of weights does not fit this num. of inputs.")
        return 1;
    else: # weights have been provided and in the correct quantity.
        w = weights
    # format for the 1D array of weights:
    # example: 100 inputs, 10 neurons => 909 weights
    #
    # ~~~~~~~~~~ inner layer ~~~~~~~~~~
    # w[0] -> w[wnum - neurons] => range(0, wnum - neurons + 1)
    #
    # w[0], w[1], w[2]... w[99] => neuron 0 => s = x0w0 + x1w1 + ... x99w99
    # w[100], w[101] ... w[199] => neuron 1 => s = x0w100 + x1w101 + ... x99w199
    # ...
    # w[800], w[801] ... w[899] => neuron 8 => s = x0w800 + x1w801 + ... x99w899
    #
    # ~~~~~~~~~~ output layer ~~~~~~~~~~
    # w[wnum - neurons + 1] -> w[wnum - 1] => range(wnum - neurons + 1, wnum)
    #
    # w[900], w[901] ... w[908] => neuron 9 => s = y0w900 + y1w901 + ... y8w908
    
    
    # start creating nodes of neurons
    nodes = []
    w_start = 0
    for n in range(0, neurons-1): # create the central layer.
        w_stop = xnum + w_start
        nodes.append(Neuron(x, w[w_start:w_stop], scale=1)) # exclusive stop, i.e.
        w_start = w_stop                           # to reach idx 899, w_stop = 900.
    y = []
    for n in range(0, neurons-1): # retrieve the outputs from the central layer.
        y.append(nodes[n].y)
    y = np.array(y)
    y = y.reshape(len(y), 1)
    nodes.append(Neuron(y, w[w_stop:wnum], scale=scale)) # create the output node.
    
    # create the neural network object
    neuralNet = NeuralNet(nodes, x, w, scale=scale)
    
    return neuralNet;

# give a neural net a question (unidentified data) and an answer,
# the neural net processes the data, provides an output, and that
# output is evaluated against the answer to determine error ('fire').
# That evaluation is then used to tweak the weights of all nodes to
# slightly different values to iteratively reduce error ('adjust').
def fire_and_adjust(Net, answer, data=[0], learning_rate=0.01, dropout=False):
    Q = answer
    nodenum = Net.nodenum
    
    # give neural net new data
    if len(data) == 1: # if true, data has not been provided, use old data.
        y = Net.y
        data = Net.x
    else:
        """data = normalize(data, scale=norm_factor*(10/nodenum))"""
        Net.set_inputs(data)
    
    # update the network
    Net.refresh()
    # check for dropout
    if dropout == True:
        Net.apply_dropout()
    y = Net.y
    y1 = y # to be returned.
    
    # calc current error amount
    E = error(y, Q)
    E1 = E
    #print(f"ans={Q} output={y:.5} error={E:.5}")
    
    # calc backpropagation from output node to inner nodes
    s = Net.neurons[-1].s
    f = sigmoid(s)
    w = Net.neurons[-1].w
    # find derivatives for output node
    dE_dyn = derive_E_dy(y, Q) # n indicates last node (output neuron)
    dy_dsn = derive_y_ds(y, f)
    ds_dxn = derive_s_dx(w)
    dE_dxn = dE_dyn * dy_dsn * ds_dxn # vector quantity, size n-1 x 1
                                      # change in error / change in input[x]
    # calc new target values for the inner nodes' outputs
    y_old = []
    for n in range(0, nodenum-1): # retrieve the outputs from the central layer.
        y_old.append(Net.neurons[n].y)
    y_old = np.array(y_old)
    y_old = y_old.reshape(len(y_old), 1)
    y_new = y_old - learning_rate * (dE_dxn) # vector quantity
    # this vector serves as the "answers" (Q) for the inner nodes.
    
    # calc backpropagation from inner layer nodes to input layer
    # error
    y = y_old
    Q = y_new
    # get values
    s = []
    f = []
    for n in range(0, nodenum-1):
        s.append(Net.neurons[n].s)
        f.append(sigmoid(s[n]))
    s = (np.array(s)).reshape(len(s), 1)
    f = (np.array(f)).reshape(len(f), 1)
    # find derivatives
    dE_dy = derive_E_dy(y, Q) # vector shape = (nodenum - 1) x 1
                              # dE[node0] / dy[node0] = dE_dy[0]
    dy_ds = derive_y_ds(y, f) # vector shape = (nodenum - 1) x 1
                              # dy[node0] / ds[node0] = dy_ds[0]
    ds_dw = derive_s_dw(data) # vector shape = (len(data)) x 1
    # to calculate dE_dw:
    # dE_dy       dy_ds     ds_dw      dE_dw
    # [node0]     [node0]   [input0]   [node0 = Δw0 Δw1 Δw2 ... ΔwM]
    # [node1] (*) [node1] * [input1] = [node1 = Δw0 Δw1 Δw2 ... ΔwM]
    # ...         ...       ...        ...
    # [nodeN]     [nodeN]   [inputM]   [nodeN = Δw0 Δw1 Δw2 ... ΔwM]
    # N x 1       N x 1     M x 1      N x M
    # hadamard ^ product producing a N x 1 vector
    #   = [dE_dy[0]*dy_ds[0] dE_dy[1]*dy_ds[1]... dE_dy[N]*dy_ds[N]]
    hadamard = dE_dy * dy_ds
    dE_dw = np.zeros((nodenum-1, len(data), 1))
    for n in range(0, nodenum-1):
        dE_dw[n][:] = (hadamard[n] * ds_dw) # 9 vectors each with
                                            # shape = (len(data)) x 1
                                            
    # find new weight values for inner nodes, set, and refresh.
    new_weights = []
    for n in range(0, nodenum-1):
        new_weights.append(Net.neurons[n].w - learning_rate * (dE_dw[0][:]))
    new_weights = np.concat(new_weights)
    Net.set_inner_weights(new_weights)
    Net.refresh() # output node has now been updated to reflect the changes.
    
    # recalculate the output node values to find new weights
    # check for dropout
    if dropout == True:
        Net.apply_dropout()
    # current error amount
    y = Net.y
    Q = answer
    E = error(y, Q)
    #print(f"ans={Q} output={y:.5} error={E:.5}")
    
    # calc backpropagation from output node to inner nodes
    s = Net.neurons[-1].s
    f = sigmoid(s)
    w = Net.neurons[-1].w
    x = Net.neurons[-1].x
    # find derivatives for output node
    # this time we are trying to find dE/dw instead of dE/dx
    dE_dyn = derive_E_dy(y, Q) # n indicates last node (output neuron)
    dy_dsn = derive_y_ds(y, f)
    ds_dwn = derive_s_dw(x)
    dE_dwn = dE_dyn * dy_dsn * ds_dwn # vector quantity, size n-1 x 1
                                      # change in error / change in outputs[y]
    
    # find new weight values for output node, set, and refresh.
    w_new = w - learning_rate * dE_dwn
    w_new = np.reshape(w_new, (nodenum-1, 1))
    Net.set_output_weights(w_new)
    Net.refresh()
    
    # current error amount
    y2 = Net.y
    E2 = error(y, Q)
    #print(f"ans={Q} output={y:.5} error={E:.5}")
    
    return Q, y1, E1, y2, E2;

# normalize a dataset to be within the scaling number.
def normalize(data, scale=1):
    mn = np.min(data)
    mx = np.max(data)
    norm_data = (data - mn) * (1 / (mx - mn)) * scale
    return norm_data;

# standardize a dataset to be within sigma=1 and mean=0
def standardize(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    standardized_data = (data - mean) / std_dev
    return standardized_data;

def generate_random_weights(count):
    #w = np.random.rand(count, 1) # random values from [0, 1) in col vector.
    w = np.random.normal(loc=0.0, scale=0.05, size=(count, 1))
    w = np.array(w)
    return w;

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
        digit = round(answers[n])
        guess = round(predictions[n])
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
# use to create exams and answer keys.
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

# ~~~~~~~~~~~~~~~ basic mathematic formulas ~~~~~~~~~~~~~~~
def sop(x, w):
    # sum-of-products of inputs with weights.
    #      n                                                        [w1]
    #  s = Σ (x[i] * w[i]) = x1*w1 + x2*w2 + xn*wn = [x1, x2, xn] . [w2]
    #     i=0                                                       [wn]
    #print(f"shape of x/w: {x.shape}/{w.shape}")
    sop = np.dot(x.T, w)
    #print(f"sop = {sop.item()}")
    return sop.item(); # dot multiplier, ensuring value is a scalar.
    
def sigmoid(s):
    #             1
    # f(s) = ------------
    #         1 + e^(-s)
    return (1.0 / (1 + np.exp(-1 * s)));

def error(y, Q):
    #      1
    # E = --- * (answer - output)^2
    #      2
    # 'output' is the neuron's prediction,
    # usually f(s) (but not always).
    return (0.5) * (y - Q)**2;

def derive_E_dy(y, Q):
    # partial derivative of error (E) w/ respect to [WRT] output (y).
    # Q = correct answer a.k.a desired value.
    #  d        1                    2
    # -- E = ( --- * (Q - y)^2 )' = --- * (Q - y) * (-1) = y - Q
    # dy        2                    2
    return y - Q;

def derive_y_ds(y, f):
    # partial derivative of output (y) WRT sum-of-products (s).
    #  d                        1                   1
    # -- y = sigmoid(s)' = ------------ * (1 - ------------) = sigmoid(s) * (1 - sigmoid(s))
    # ds                    1 + e^(-s)          1 + e^(-s)
    # if y = sigmoid(s) * scaling_parameter, then dy = y * (1 - sigmoid(s))
    return y * (1 - f);
    
def derive_s_dw(x):
    # partial derivative of sum-of-products (s) WRT weights (w).
    #  d                                                    d
    # ---  s = (x1*w1 + x2*w2 + xn*wn)' = x1 + 0 + 0 --->> -- s = [x1 x2 xn]
    # dw1                                                  dw
    return x;

def derive_s_dx(w):
    # partial derivative of sum-of-products (s) WRT inputs (x).
    #  d                                                    d
    # ---  s = (x1*w1 + x2*w2 + xn*wn)' = w1 + 0 + 0 --->> -- s = [w1 w2 wn]
    # dx1                                                  dx
    return w;

"""def error_predicted_deriv(predicted, target):
    return 2*(predicted-target)

def sigmoid_sop_deriv(sop):
    return sigmoid(sop)*(1.0-sigmoid(sop))

def sop_w_deriv(x):
    return x

def update_w(w, grad, learning_rate):
    return w - learning_rate*grad"""
    
    
    
    
    
    
    
    
    
    