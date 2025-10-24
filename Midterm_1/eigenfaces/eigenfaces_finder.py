# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 13:11:03 2025

@author: S.W. Marsden
"""

# Methods referenced from:
# onionesquereality.wordpress.com/2009/02/11/face-recognition-using-eigenfaces-and-distance-classifiers-a-tutorial/
# NOTE: greek letters are hard,
#       G = gamma,  S = psi,  F = phi,  etc.

# M = number of images in the dataset.
# K = number of most significant Eigenfaces that can approximate a face.
# K < M

# Nc = number of values in 1 row of an image.
# Nr = number of values in 1 column of an image.
# if an image is a matrix of Nc x Nr, then that same image can be represented
# by a single vector of size (Nc) * (Nr)

# ~~~~~~~~~~ LIBRARIES ~~~~~~~~~~
import os
import numpy as np
import skimage as ski
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy import ndimage as scipy


# ~~~~~~~~~ DEFINITIONS ~~~~~~~~~
REDUCE = True
IMAGE_FOLDER = 'K:/Documents/Python (Spyder)/Midterm 1/pain_crops'
MAX_IMAGES = 9
PLOT_GRID = (3, 3)
PLOC = np.empty( ((PLOT_GRID[0]*PLOT_GRID[1]), 2) ).astype(np.uint8)
"""
PLOC[0] = (1,1); PLOC[1] = (1,2); PLOC[2] = (1,3)
PLOC[3] = (2,1); PLOC[4] = (2,2); PLOC[5] = (2,3)
"""

class Image:
    def __init__(self, name, data, ploc):
        self.name = name
        self.data = data
        self.ploc = ploc
        
    def easy_plot_info(self):
        return self.data, self.ploc, self.name


# ~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~
import eigenfaces_functions as eff


# ~~~~~~~~~~ MAIN CODE ~~~~~~~~~~
# setup for plotting multiple images
plt.rc('font', size=6)
fig, ax = plt.subplots(PLOT_GRID[0], PLOT_GRID[1],
                       dpi=1200, constrained_layout=True)

# setup plot location tuples (coordinates)
idx = 0
for row in range(0, PLOT_GRID[0]):
    for col in range(0, PLOT_GRID[1]):
        PLOC[idx] = (row+1, col+1)
        idx += 1
idx = 0

# Step 1: read M images I[0], I[2], I[3] ... I[M].
I = []
I_filenames = []
idx = 0
print("Reading images from ", IMAGE_FOLDER)
for filename in os.listdir(IMAGE_FOLDER):
    if idx < MAX_IMAGES:
        file_path = os.path.join(IMAGE_FOLDER, filename)
        img = iio.imread(file_path)
        # convert to grayscale, if needed, to simplify analysis
        if len(img.shape) == 3:
            img = ski.color.rgb2gray(img)
        # reduce in size, if needed, to speed up processing time
        if REDUCE == True:
            img = scipy.zoom(img, 0.5)
        img = (255 * img).astype(np.uint8)
        I.append(img)
        I_filenames.append(filename)
        idx += 1
    else:
        break
print(f"{idx} images read.")
print("Image size: ", I[0].size)
idx = 0

# add images to list of image objects
img_list = []
for i in range(0, MAX_IMAGES):
    new_img = Image(I_filenames[i], I[i], PLOC[i])
    img_list.append(new_img)

# display the entire list of image objects
for i in range(0, MAX_IMAGES):
    eff.ezplot(img_list[i].easy_plot_info(), ax)
    

# Step 2: represent each image I[i] as a vector G[i].
# e.g.
#                                          [ a ]
#         [a b c]                          [ b ]
# I[49] = [d e f] = concatenate => G[49] = [...]
#         [g h i]                          [ h ]
#                                          [ i ]
G = []
for i in range(0, MAX_IMAGES):
    count = 0
    pixel_vector = []
    for y in range(I[i].shape[0]):
        for x in range(I[i].shape[1]):
            new_pixel = I[i][y][x]
            pixel_vector.append(new_pixel)
            count += 1
    G.append(pixel_vector)
G = np.array(G)

# verify that vectorization step went as planned,
# check that the last value of the last image matches the last G vector and
# that the pixel count of the image matches the length of the vector.
if I[-1][-1][-1] == G[-1][-1] and I[-1].size == len(G[-1]):
    print("Images vectorized.")
else:
    print("WARNING in image vectorization.")


# Step 3: find the average face vector S.
#               M
# S = (1 / M) SIGMA{ G[i] }
#              i=1
S = np.zeros(len(G[0]))
for i in range(0, MAX_IMAGES):
    S = S + G[i]
S = S.astype(float)
S = np.round((1/MAX_IMAGES) * S)
S = S.astype(np.uint8)
if len(S) == len(G[-1]):
    print("Average face vector calculated.")
else:
    print("WARNING in face vector average.")
    

# Step 4: subtract the avg. face from each face vector G[i] to get
# the set of vectors F[i]. This removes "common" features of the faces.
# F[i] = G[i] - S
F = []
for i in range(0, MAX_IMAGES):
    unique_features = G[0] - S
    F.append(unique_features)
if I[-1].size == F[-1].size:
    print("Common features removed from faces.")
else:
    print("WARNING in common features removal.")

# Step 5: find the covariance matrix C.
# C = A(A^T) where A = [ F[0] F[1] F[2] ... F[M] ]
# NOTE: C is a N^2 * N^2 matrix and A is a N^2 * M matrix.

# Step 6: calculate the Eigenvectors U[i] of C.
# BUT this calculation would return N^2 Eigenvectors each being N^2, TOO BIG!

# Step 7: instead consider (A^T)A. Finding the Eigenvectors of this
# yields a more manageable M Eigenevectors each being M * 1.
# these are the Eigenvectors V[i].
matrixA = np.zeros((len(F), MAX_IMAGES))
for col in range(0, MAX_IMAGES):
    for row in range(0, len(F)):
        matrixA[row][col] = F[col][row]
C = (matrixA.T) * matrixA
if C.size == (I[-1].size)**2: # does covariance size equals (N^2)*(N^2)?
    print("Covariance matrix calculated.")
else:
    print("WARNING in covariance matrix calc.")

# Step 8: find the best M Eigenvectors of C = A(A^T).
# matrix math tells us that U[i] = A(V[i]) and that ||U[i]|| = 1.

# Step 9: select the best K Eigenvectors by trial-and-error.
# when converted back into image form (i.e. seperated back into a 2D array)
# these vectors U[i] become the Eigenfaces for this dataset.
# now each face in the dataset (minus the mean), F[i] can be represented as
# a linear combination of these Eigenfaces.
# (and adding the mean back in yields the original face?)
#  but we still need the weights.
#          K
# F[i] = SIGMA{ W[j] U[j] }
#         j=1

# Step 10: find the weights W[j] of each Eigenvector.
# W[j] = ( U[j]^T ) F[i]
# each image in the dataset has a whole vector of weights belonging to it
#                     [ W[0] ]
# so therefore Q[i] = [ ...  ] where i = 0, 1, 2 ... M.
#                     [ W[k] ]
