# ~~~~~~~~~~ LIBRARIES ~~~~~~~~~~
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import skimage as ski
from scipy import ndimage as scipy
import cv2

# ~~~~~~~~~ DEFINITIONS ~~~~~~~~~
MAX_PIXELS = 500 * 500
PLOT_GRID = (2, 3)
PLOC1 = (1,1); PLOC2 = (1,2); PLOC5 = (1,3);
PLOC3 = (2,1); PLOC4 = (2,2); PLOC6 = (2,3);

# ~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~
# display images and figures on 2D grid of plots with name ax[][]
def easy_plot(dataIn, plotLoc, name='', plotType='image', colors='gray', gridshare=0):
    rowI = plotLoc[0] - 1
    colI = plotLoc[1] - 1
    if plotType == 'image':
        ax[rowI][colI].imshow(dataIn, cmap=colors)
        ax[rowI][colI].set_title(name)
    if plotType == 'pointOverlay':
        ax[rowI][colI].autoscale(False)
        ax[rowI][colI].plot(dataIn[:, 1], dataIn[:, 0], colors, markersize=3)
    if gridshare == 1:
        ax[rowI][colI].sharex(ax[0][0])
        ax[rowI][colI].sharey(ax[0][0])
    
    return 0;

# Compute and show the edge magnitude and direction for synthetic and corrupted images.
def calculateEdges(dataIn):
    img = dataIn
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    # Convolve images with the Sobels.
    gradX = np.array(scipy.convolve(img, sobel_x))
    gradY = np.array(scipy.convolve(img, sobel_y))
    
    # magnitude = distance of gradient
    magArr = np.sqrt(gradX**2 + gradY**2)
    
    return magArr


# ~~~~~~~~~~ MAIN CODE ~~~~~~~~~~
# setup for plotting multiple images
plt.rc('font', size=6)
fig, ax = plt.subplots(PLOT_GRID[0], PLOT_GRID[1], dpi=600, constrained_layout=True)

# read, modify, and plot original (OG) image
imgOG = iio.imread('CoffeeBeans.tif')
pixel_count = imgOG.shape[0] * imgOG.shape[1]
# convert to grayscale, if needed, to simplify analysis
#if imgOG.shape[2] > 0:
imgOG = ski.color.rgb2gray(imgOG)
# reduce size of image, if needed, to fit max pixels defined
while pixel_count > MAX_PIXELS:
    print("Too many pixels! Reducing from ", pixel_count)
    #downscale_factor = round( (MAX_PIXELS / pixel_count ), 4)
    imgOG = scipy.zoom(imgOG, 0.5)
    pixel_count = imgOG.shape[0] * imgOG.shape[1]
easy_plot(imgOG, PLOC1, 'Original Image')
imgNEW = imgOG

# noise removal
imgOLD = imgNEW
imgBlur = cv2.GaussianBlur(imgOLD, (5,5), 0)
imgNEW = imgBlur
easy_plot(imgNEW, PLOC3, 'Step 1, blur')

# binarize image
imgOLD = imgNEW
thresh = ski.filters.threshold_otsu(imgOLD)
imgBinary = imgOLD < thresh
#thresholds = ski.filters.threshold_multiotsu(imgOG) # creates three regions
#imgBinary = np.digitize(imgOLD, bins=2)
imgNEW = imgBinary.astype(np.uint8)
easy_plot(imgNEW, PLOC2, 'Step 2, binarize')

# distance transform with maximum points noted
imgOLD = imgNEW
imgDistTrans = scipy.distance_transform_edt(imgOLD)
imgNEW = imgDistTrans
easy_plot(imgNEW, PLOC4, 'Step 3, distance transform')
# find the markers (low point of the basins)
lowPoints = ski.feature.peak_local_max(imgDistTrans, min_distance=9, exclude_border=0)
#lowPoints = ski.feature.peak_local_max(imgDistTrans, footprint=np.ones((5, 5)))
easy_plot(lowPoints, PLOC4, plotType='pointOverlay', colors='r.')
print("Coffee bean count 1: ", int(lowPoints.size/2))

# watershed segmentation and labeling
imgOLD = imgNEW
lowPointImg = np.zeros(imgDistTrans.shape, dtype=bool) # convert to mask image
lowPointImg[tuple(lowPoints.T)] = True # assign coordinates in mask image = true
labels, count = scipy.label(lowPointImg)
imgSegments = ski.segmentation.watershed(-imgDistTrans, labels, mask=imgBinary)
imgNEW = imgSegments
easy_plot(imgNEW, PLOC5, 'Step 4, watershed flooding', colors=plt.cm.nipy_spectral_r)
print("Coffee bean count 2: ", int(count))
print(lowPoints.shape)
easy_plot(lowPoints, PLOC5, plotType='pointOverlay', colors='w.')
ax[PLOC5[0]-1][PLOC5[1]-1].set_axis_off()
for n in range(lowPoints.shape[0]):
    ax[PLOC5[0]-1][PLOC5[1]-1].annotate(n+1, xy=(lowPoints[n][1],lowPoints[n][0]),
                                        textcoords='data',
                                        horizontalalignment='left',
                                        verticalalignment='top',
                                        color='black')