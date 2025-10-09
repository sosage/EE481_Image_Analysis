# ~~~~~~~~~~ LIBRARIES ~~~~~~~~~~
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from skimage import measure, filters, color as ski
from scipy import ndimage as sp

# ~~~~~~~~~ DEFINITIONS ~~~~~~~~~
MAX_PIXELS = 500 * 500

# ~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~
# display images and figures on 2D grid of plots with name ax[][]
def easy_plot(dataIn, plotLoc, name='', plotType='image', colors='gray', gridshare=0):
    rowI = plotLoc[0] - 1
    colI = plotLoc[1] - 1
    if plotType == 'image':
        ax[rowI][colI].imshow(dataIn, cmap=colors)
        ax[rowI][colI].set_title(name)
    if gridshare == 1:
        ax[rowI][colI].sharex(ax[0][0])
    
    return 0;

# ~~~~~~~~~~ MAIN CODE ~~~~~~~~~~
# setup for plotting multiple images
plt.rc('font', size=6)
fig, ax = plt.subplots(2, 3, dpi=600, constrained_layout=True)

# read original (OG) image
imgOG = iio.imread("webCoffeeBeans.jpg")
pixel_count = imgOG.shape[0] * imgOG.shape[1]
# convert to grayscale, if needed, to simplify analysis
if imgOG.shape[2] > 0:
    imgOG = ski.rgb2gray(imgOG)
# reduce size of image, if needed, to fit max pixels defined
while pixel_count > MAX_PIXELS:
    print(pixel_count)
    #downscale_factor = round( (MAX_PIXELS / pixel_count ), 4)
    imgOG = sp.zoom(imgOG, 0.5)
    pixel_count = imgOG.shape[0] * imgOG.shape[1]
ploc1 = (1, 1)
easy_plot(imgOG, ploc1, 'Original Image')
