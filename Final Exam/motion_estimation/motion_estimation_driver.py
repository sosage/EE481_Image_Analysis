"""
Image Analysis - Final Exam - Question 2 - Motion Estimation
Driver File
Created on Tue Dec  9 15:26:00 2025

@author: S.W. Marsden
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import imageio.v3 as iio
import matplotlib.pyplot as plt

import motion_functions as skyler

class Image:
    def __init__(self, name, data, ploc):
        self.name = name
        self.data = data
        self.ploc = ploc
        
    def ezplot_info(self):
        return self.data, self.ploc, self.name;

# setup the multiple plots for output visualization.
plot_rows = 2
plot_cols = 4
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
patch = (3, 3) # the neighborhood around a pixel to be
               # considered to have constant flow
               # i.e. where Ix * u + Iy * v + It = 0 is true.
PA1 = sliding_window_view(IMG[0].data, patch) # divide each img
PA2 = sliding_window_view(IMG[1].data, patch) # into patches.
# take each patch and apply the LK method to find the resp-
# ective motion vector x̂ = [u v]. 1 vector per patch.
It = PA1 - PA2 # temporal derivative (change over time) for LK.
# test case:
# u, v = skyler.compute_lucas_kanade_motion(PA1[100][100], It[100][100])
"""motion_v1 = np.zeros((IMG[0].data.shape[0],
                      IMG[0].data.shape[1], 2))
motion_v2 = np.zeros((IMG[1].data.shape[0],
                      IMG[1].data.shape[1], 2))"""

motion_v1 = np.zeros((PA1.shape[0], PA1.shape[1], 2))
"""for row in range(0, PA1.shape[0]):
    for col in range(0, PA1.shape[1]):
        #print(row, col)
        motion_v1[row,col] = skyler.compute_LuKan(PA1[row,col],
                                                   It[row,col])
motion1 = (motion_v1[:,:,0]**2 + motion_v1[:,:,1]**2)**(1/2)"""
motion1 = motion_v1[:,:,0] # TEST CASE, REMOVE
IMG.append(Image('mag. of motion, F1', # img [2]
                 motion1, PLOC[2]))
skyler.ezplot(IMG[2].ezplot_info(), ax)

motion_v2 = np.zeros((PA2.shape[0], PA2.shape[1], 2))
for row in range(0, PA2.shape[0]):
    for col in range(0, PA2.shape[1]):
        #print(row, col)
        motion_v2[row,col] = skyler.compute_LuKan(PA2[row,col],
                                                   It[row,col])
motion2 = 255*(motion_v2[:,:,0]**2 + motion_v2[:,:,1]**2)**(1/2)
for row in range(0, PA2.shape[0]):
    for col in range(0, PA2.shape[1]):
        if motion2[row,col] > 255:
            motion2[row,col] = 255
print(np.max(motion2), np.min(motion2))
IMG.append(Image('mag. of motion, F2', # img [3]
                 motion2, PLOC[3]))
skyler.ezplot(IMG[3].ezplot_info(), ax)
