# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:26:32 2025
@author: S.W. Marsden
"""
# Reference: https://www.youtube.com/watch?v=G39dVoiivZk
# from skimage.restoration import denoise_tv_chambolle
# usage:
#           denoise_tv_chambolle(img,
#                                weight=0.1, more weight = more denoising = less image fidelity
#                                eps=0.0002, rel. diff. of value of the "cost function" (optional)
#                                n_iter_max=200, no. of iterations (optional)
#                                multichannel=False) RGB image or greyscale

import numpy as np
import skimage as ski
import imageio.v3 as iio
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle

# ~~~~~~~~~ DEFINITIONS ~~~~~~~~~
PLOT_GRID = (3, 4)
PLOC = np.empty( ((PLOT_GRID[0]*PLOT_GRID[1]), 2) ).astype(np.uint8)
idx = 0
for row in range(0, PLOT_GRID[0]):
    for col in range(0, PLOT_GRID[1]):
        PLOC[idx] = (row+1, col+1)
        idx += 1
idx = 0
# Plot location coordinates. For a 3x3 grid of plots, there are 9 locations
# each with a 2-element coordinate (row, col).
# PLOC[0] = (1,1); PLOC[1] = (1,2); PLOC[2] = (1,3)
# PLOC[3] = (2,1); PLOC[4] = (2,2); PLOC[5] = (2,3)
# PLOC[6] = (3,1); PLOC[7] = (3,2); PLOC[8] = (3,3)

class Image:
    def __init__(self, name, data, ploc):
        self.name = name
        self.data = data
        self.ploc = ploc
        
    def ezplot_info(self):
        return self.data, self.ploc, self.name;

# setup for plotting multiple images
def plotArray_init():
    plt.rc('font', size=6)
    fig, ax = plt.subplots(PLOT_GRID[0], PLOT_GRID[1],
                           dpi=1200, constrained_layout=True)
    return ax;

# grayscale 2, 3, or 4 channel images
def Grayout(img):
    img_dimension = len(img.shape)
    if img_dimension == 2:
        return img;
    elif img_dimension == 3:
        img = ski.color.rgb2gray(img[:,:,0:3])
        # if there are more than 3 color params (RGBA or CMYK),
        # only 0 thru 3 are used.
        return img;
    else:
        print("ERROR while grayscaling.")
        return 1;

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
    
    if plotType == 'pointOverlay':
        ax[rowI][colI].autoscale(False)
        ax[rowI][colI].plot(data[:, 1], data[:, 0], colors,
                            markeredgewidth=0.25, markersize=3,
                            markerfacecolor='none')
    
    if gridshare == 1:
        ax[rowI][colI].sharex(ax[0][0])
        ax[rowI][colI].sharey(ax[0][0])
   
    return 0;

# denoise the image using the total variation chambolle method
def totalVar_denoise(img, intensity=0.5, loops=100):
    denoised_img = denoise_tv_chambolle(img, weight=intensity, eps=0.0002,
                                        max_num_iter=loops)
    return denoised_img;

# take a square section from top left of an image for detail analysis
def snip(img, sizeX=300, sizeY=300):
    img_snip = img[:sizeY,:sizeX]
    return img_snip;
    

# ~~~~~~~~~~ MAIN CODE ~~~~~~~~~~
ax = plotArray_init()
Cryo1 = iio.imread('Cryo_EM_Image1.jpeg')
Cryo2 = iio.imread('Cryo_EM_Image2.png')
# convert to grayscale, if needed, to simplify analysis
Cryo1 = Grayout(Cryo1)
Cryo2 = Grayout(Cryo2)
# create a list of image objects for plotting
IMG = []
IMG.append(Image("Cryo 1", Cryo1, PLOC[0])) #IMG[0]
IMG.append(Image("Cryo 2", Cryo2, PLOC[8])) #IMG[1]
# apply total variation denoising
IMG.append(Image("C1, 3 iterations", totalVar_denoise(Cryo1, loops=3), PLOC[1])) #IMG[2]
IMG.append(Image("C2, 3 iterations", totalVar_denoise(Cryo2, loops=3), PLOC[9])) #IMG[3]
IMG.append(Image("C1, 6 iterations", totalVar_denoise(Cryo1, loops=6), PLOC[2])) #IMG[4]
IMG.append(Image("C2, 6 iterations", totalVar_denoise(Cryo2, loops=6), PLOC[10])) #IMG[5]
IMG.append(Image("C1, 9 iterations", totalVar_denoise(Cryo1, loops=9), PLOC[3])) #IMG[6]
IMG.append(Image("C2, 9 iterations", totalVar_denoise(Cryo2, loops=9), PLOC[11])) #IMG[7]
# take a snip of a small section of Cryo1 to see better detail
IMG.append(Image("C1, zoomed-in", snip(IMG[0].data), PLOC[4])) #IMG[8]
IMG.append(Image("C1, 3 iter, zoomed-in", snip(IMG[2].data), PLOC[5])) #IMG[9]
IMG.append(Image("C1, 6 iter, zoomed-in", snip(IMG[4].data), PLOC[6])) #IMG[10]
IMG.append(Image("C1, 9 iter, zoomed-in", snip(IMG[6].data), PLOC[7])) #IMG[11]
# plot all images in the IMG list
for i in IMG:
    ezplot(i.ezplot_info(), ax)
