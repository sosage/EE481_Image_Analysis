# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:28:31 2025

@author: S.W. Marsden
"""

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

# ~~~~~~~~~~ Easy Plot ~~~~~~~~~~
# plot an array of data or an image on a multiplot layout
def ezplot(plotInfo, plotObj, plotType='image', colors='gray', gridshare=0):
    # plotInfo is a 3 wide array with data, ploc, and name.
    rowI = plotInfo[1][0].astype(int) - 1 # plotInfo[1] is ploc e.g. (row 2, col 4)
    colI = plotInfo[1][1].astype(int) - 1 
    data = plotInfo[0]
    name = plotInfo[2]
    ax = plotObj
    
    if plotType == 'image':
        ax[rowI][colI].imshow(data, cmap=colors)
        ax[rowI][colI].set_title(name)
        ax[rowI][colI].set_xticks([])
        ax[rowI][colI].set_yticks([])
    
    if plotType == 'pointOverlay':
        ax[rowI][colI].autoscale(False)
        ax[rowI][colI].plot(data[:, 1], data[:, 0], colors, markeredgewidth=0.25,
                            markersize=3, markerfacecolor='none')
    
    if gridshare == 1:
        ax[rowI][colI].sharex(ax[0][0])
        ax[rowI][colI].sharey(ax[0][0])
   
    return 0;

# ~~~~~~~~~~ Matrix to Vector ~~~~~~~~~~
def m2v():
    
    return 0;

# ~~~~~~~~~~ Vector to Matrix ~~~~~~~~~~
def v2m(vect, matrixShape):
    height = matrixShape[0]
    width = matrixShape[1]
    mtrx = np.zeros((height, width))
    
    i = 0
    for row in range(0, height):
        for col in range(0, width):
            mtrx[row][col] = vect[i]
            i += 1
    
    return mtrx;