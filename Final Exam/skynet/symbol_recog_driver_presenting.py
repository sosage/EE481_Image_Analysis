"""
Image Analysis - Final Exam - Question 4 - Character ID'ing
from MNIST Dataset Using Neural Network, Driver File
Created on Fri Dec  12 13:47:00 2025

@author: S.W. Marsden
"""

#import imageio.v3 as iio

import numpy as np
import symbol_recog_funcs as skyler
import skynet

class Image:
    def __init__(self, name, data, ploc):
        self.name = name
        self.data = data
        self.ploc = ploc
        
    def ezplot_info(self):
        return self.data, self.ploc, self.name;

# global variables
plot_rows = 2
plot_cols = 2
iters1 = 500

# setup the multiple plots for output visualization.
plotGrid = (plot_rows, plot_cols)
PLOC = skyler.initialize_plot_locations(plotGrid)
ax = skyler.initialize_multiplot(plotGrid, 
                                 text_size=6,
                                 res=800)

# read in the initial images, convert to grayscale,
# and add to Image list.
TRAIN = []
TEST = []

train_lbls = skyler.load_idx1_ubyte('MNISTdataset', 'train-labels.idx1-ubyte')
train_imgs = skyler.load_idx3_ubyte('MNISTdataset', 'train-images.idx3-ubyte')
test_lbls = skyler.load_idx1_ubyte('MNISTdataset', 't10k-labels.idx1-ubyte')
test_imgs = skyler.load_idx3_ubyte('MNISTdataset', 't10k-images.idx3-ubyte')

"""
for i in range(0, train_imgs.shape[0]):
    TRAIN.append(Image(f"train{i}({train_lbls[i]})", train_imgs[i], PLOC[0]))

for i in range(0, test_imgs.shape[0]):
    TEST.append(Image(f"test{i}({test_lbls[i]})", test_imgs[i], PLOC[0]))
    
for i in range(0, 25):
    if i < 20:
        TRAIN[i].ploc = PLOC[i]
        skyler.ezplot(TRAIN[i].ezplot_info(), ax)
    else:
        TEST[i-20].ploc = PLOC[i]
        skyler.ezplot(TEST[i-20].ezplot_info(), ax)
"""
print(f"Read {len(train_lbls)} training imgs + {len(test_lbls)} test imgs from MNIST.")

sample_idx = 0
sample_img = train_imgs[sample_idx,:,:]
#sample_img = np.ones((28,28))*100
#sample_img[12,12] = 0
#print(sample_img.size) -> 784
print(f"Sample image label: {train_lbls[sample_idx]}")
samp_vect = np.array(sample_img.reshape(784, 1))

# smart (?) architecture
# x 000 -\ 
# x 001 -\\     /- node 0 --\
# x 002 -\\\   //- node 1 --\\
# .       ------   ...     node N
# .       /    \   ...       /
# .      /      \  node N-1 /
# x 783 /
# all x inputs connect to all nodes 0 thru N-1
# outputs of nodes 0 thru N-1 connect to node N.
"""samp_vect = skynet.normalize(samp_vect, scale=0.1) # scale here helps downsize the
                                                   # large number of inputs to prevent
                                                   # saturating the sigmoid(s) calc."""
samp_vect = skynet.normalize(samp_vect)
neuralNet1 = skynet.create_neuralNet(inputs=samp_vect, neurons=10, scale=9.0)
Q0, y01, E01, y02, E02 = skynet.fire_and_adjust(neuralNet1, train_lbls[sample_idx]) # using inputs from net init.
print(f"Session #0: Q={Q0} y1={y01:.4} E1={E01:.4} y2={y02:.4} E2={E02:.4}")

# conduct test identifying the image labeled "5" comparing the 2 networks
gens = 1000
found = 0
E1s = []
for n in range(0, gens):
    Q, y1, E1, y2, E2 = skynet.fire_and_adjust(neuralNet1,
                                               train_lbls[sample_idx],
                                               samp_vect,
                                               dropout=True)
    E1s.append(E1)
    if E2 < 8e-5 and found == 0:
        found = n

# dumb architecture
# x 000 \
# x 001  \
# x 002   \
# .        \
# .         ---> node 0 ---> y
# .        /
# x 781   /
# x 782  /
# x 783 /

# prepare 1 img for processing (img label "5")
sample_idx = 0
sample_img = train_imgs[sample_idx,:,:]
print(f"Sample image label: {train_lbls[sample_idx]}")
samp_vect = np.array(sample_img.reshape(784, 1))
samp_vect = skynet.normalize(samp_vect)

# run the img thru Dumb-net 1000 times to find its predictions and errors.
gens = 1000
init_pred, init_err, init_w, init_found = skyler.build_net(samp_vect, train_lbls[0],
                                                           gens)

# plot
x0_axis = np.arange(0,gens,1).tolist()
y0_axis = init_err
skyler.ezplot((([x0_axis, y0_axis]), PLOC[0], f"img '5': dumbNet= {init_found} gens to find answer"),
              ax, plotType='plot')


skyler.ezplot((([x0_axis, E1s]), PLOC[1], f"img '5': smartNet= {found} gens to find answer"), ax, plotType='plot')

# conduct test identifying random images from the training database
train_order = skyler.generate_data_order(60000, 60000)

gens = 1
imgs = 5000
x0_axis = np.arange(0,gens*imgs,1).tolist()
found = 0
E1s = []
avgErr = 0
total_runs = 0
Qs = []
y1s = []
for img in range(0, imgs):
    sample_idx = train_order[img]
    samp_vect = train_imgs[sample_idx].reshape(784, 1)
    """samp_vect = skynet.normalize(samp_vect, scale=0.1)"""
    samp_vect = skynet.normalize(samp_vect)
    if img == 0:
        neuralNet2 = skynet.create_neuralNet(inputs=samp_vect, neurons=10, scale=9.0)
    for n in range(0, gens):
        total_runs += 1
        Q, y1, E1, y2, E2 = skynet.fire_and_adjust(neuralNet2,
                                                   train_lbls[sample_idx],
                                                   samp_vect,
                                                   learning_rate=0.05,
                                                   dropout=True)
        avgErr = ((avgErr * (total_runs - 1)) + E1) * (1/total_runs)
        E1s.append(avgErr)
        Qs.append(Q)
        y1s.append(y1)
        
        if avgErr < 8e-5 and found == 0:
            found = n

Qs = np.array(Qs)
y1s = np.array(y1s)

skyler.ezplot((([x0_axis, E1s]), PLOC[3], f"{imgs} random imgs -> smartNet w/ 50% dropout, avg error"), ax, plotType='plot')

conMat_net2, precision, recall = skynet.find_confusion(y1s, Qs)

"""
weights = skynet.generate_random_weights(784)
train_order = skyler.generate_data_order(60000, 1000)
for img in range(0, 500):
    img_idx = train_order[img]
    training_vector = train_imgs[img_idx].reshape(784, 1)
    train_predictions, training_errors, weights, found = skyler.build_net(training_vector,
                                                                          train_lbls[img_idx],
                                                                          iters=5,
                                                                          init_weights=weights)
"""
    
#test_order = skyler.generate_data_order(10000, 1000)
#test_pred, test_answers, test_err = skyler.test_net(test_imgs, test_lbls, test_order, weights)


"""random_predictions = np.zeros(10000, dtype=np.uint8)
for i in range(0, 10000):
    random_predictions[i] = random.randrange(10)
"""


# evaluate the performance of the bots by comparing their predictions to
# the actual labels (guesses vs. answer-key), generating a confusion matrix,
# and finding the average precision and recall (low false(+)/(-)'s vs. low missed true(+)'s).
#confusion_matrix, prec, rec = skyler.find_confusion(random_predictions, test_lbls)
#print(f"\nRandom predictions:   {np.sum(confusion_matrix[:,:])} predictions tested. Prec = {(np.mean(prec)*100):.2f}%, Rec = {(np.mean(rec)*100):.2f}%")

"""
    skyler.ezplot(IMG[i].ezplot_info(), ax)
    print(f"{test_lbls[i]} ", end='')
"""
#IMG.append(Image('dataset1', # img [0]
#                 skyler.convert_to_gray(img1),
#                 PLOC[0]))