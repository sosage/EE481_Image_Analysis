"""
Image Analysis - Final Exam - Question 2 - Motion Estimation
Driver File
Created on Tue Dec  9 15:26:00 2025

@author: S.W. Marsden
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import imageio.v3 as iio
#import matplotlib.pyplot as plt

import motion_functions as skyler

class Image:
    def __init__(self, name, data, ploc):
        self.name = name
        self.data = data
        self.ploc = ploc
        
    def ezplot_info(self):
        return self.data, self.ploc, self.name;

# setup the multiple plots for output visualization.
plot_rows = 3
plot_cols = 2
plotGrid = (plot_rows, plot_cols)
PLOC = skyler.initialize_plot_locations(plotGrid)
ax = skyler.initialize_multiplot(plotGrid, text_size=8, res=800)

# read in the initial images, convert to grayscale,
# add to Image list, and display.
IMG = []
img1 = iio.imread('Motion10.jpg')
img2 = iio.imread('Motion11.jpg')
IMG.append(Image('Frame 1', # img [0]
                 skyler.convert_to_gray(img1), PLOC[0]))
IMG.append(Image('Frame 2', # img [1]
                 skyler.convert_to_gray(img2), PLOC[1]))
skyler.ezplot(IMG[0].ezplot_info(), ax)
skyler.ezplot(IMG[1].ezplot_info(), ax)

# compute motion using Lucas-Kanade method for optical flow.
patch = (11, 11) # the neighborhood around a pixel to be
               # considered to have constant flow
               # i.e. where Ix * u + Iy * v + It = 0 is true.
pWidth = patch[0]
PA1 = sliding_window_view(IMG[0].data,               # divide each image into
                          patch)[::pWidth, ::pWidth] # non-overlaping slid-
PA2 = sliding_window_view(IMG[1].data,               # ing windows (patches)
                          patch)[::pWidth, ::pWidth] # giving PatchArray1 & 2.
center_offset = pWidth // 2
centers = []
for row in range(PA1.shape[0]):                   # generate a set of coords
    for col in range(PA1.shape[1]):               # for the center of each
        row_center = row * pWidth + center_offset # patch to be used later.
        col_center = col * pWidth + center_offset
        centers.append((row_center, col_center))
centers = np.array(centers)

# take each patch and apply the LK method to find the
# respective motion vector x̂ = [u v]. 1 vector per patch.
It = PA1 - PA2 # temporal derivative (change over time) for LK.

frame1_u = np.zeros((PA1.shape[0], PA1.shape[1]))
frame1_v = np.zeros((PA1.shape[0], PA1.shape[1]))
for row in range(0, PA1.shape[0]):
    for col in range(0, PA1.shape[1]):
        #print(row, col)
        frame1_u[row,col], frame1_v[row,col] = (skyler.compute_LuKan
                                                         (PA1[row,col],
                                                          It[row,col]))

skyler.ezplot((IMG[0].data, PLOC[2], 'opt. flow, frame1'), ax)
C1 = np.hypot(frame1_u, frame1_v)
ax[PLOC[2,0]-1, PLOC[2,1]-1].quiver(centers[:,1], centers[:,0],
                                    frame1_u, frame1_v, C1, cmap='viridis',
                                    pivot='mid', scale=10, width=0.004)

skyler.ezplot((IMG[0].data, PLOC[4], 'mag(opt. flow), frame1'), ax)
ax[PLOC[4,0]-1, PLOC[4,1]-1].imshow(C1)

frame2_u = np.zeros((PA2.shape[0], PA2.shape[1]))
frame2_v = np.zeros((PA2.shape[0], PA2.shape[1]))
for row in range(0, PA2.shape[0]):
    for col in range(0, PA2.shape[1]):
        #print(row, col)
        frame2_u[row,col], frame2_v[row,col] = (skyler.compute_LuKan
                                                         (PA2[row,col],
                                                          It[row,col]))

skyler.ezplot((IMG[1].data, PLOC[3], 'opt. flow, frame2'), ax)
C2 = np.hypot(frame2_u, frame2_v)
ax[PLOC[3,0]-1, PLOC[3,1]-1].quiver(centers[:,1], centers[:,0],
                                    frame2_u, frame2_v, C2, cmap='viridis',
                                    pivot='mid', scale=10, width=0.004)

skyler.ezplot((IMG[1].data, PLOC[5], 'mag(opt. flow), frame2'), ax)
ax[PLOC[5,0]-1, PLOC[5,1]-1].imshow(C2)