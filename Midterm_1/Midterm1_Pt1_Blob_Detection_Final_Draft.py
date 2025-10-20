# ~~~~~~~~~~ LIBRARIES ~~~~~~~~~~
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import skimage as ski
from scipy import ndimage as scipy
import cv2

# ~~~~~~~~~ DEFINITIONS ~~~~~~~~~
REDUCE = False
MAX_PIXELS = 500 * 500
PLOT_GRID = (2, 3)
PLOC1 =(1,1); PLOC2 =(1,2); PLOC3 =(1,3);
PLOC4 =(2,1); PLOC5 =(2,2); PLOC6 =(2,3);
PLOC7 =(3,1); PLOC8 =(3,2); PLOC9 =(3,3);
PLOC10=(4,1); PLOC11=(4,2); PLOC12=(5,3);
MANUAL_COUNT = 110

class Image:
    def __init__(self, name, data, ploc):
        self.name = name
        self.data = data
        self.ploc = ploc
        
    def easy_plot_info(self):
        return self.data, self.ploc, self.name


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
        ax[rowI][colI].plot(dataIn[:, 1], dataIn[:, 0], colors, markeredgewidth=0.25,
                            markersize=4, markerfacecolor='none')
    if gridshare == 1:
        ax[rowI][colI].sharex(ax[0][0])
        ax[rowI][colI].sharey(ax[0][0])
    
    return 0;


# ~~~~~~~~~~ MAIN CODE ~~~~~~~~~~
# setup for plotting multiple images
plt.rc('font', size=6)
fig, ax = plt.subplots(PLOT_GRID[0], PLOT_GRID[1], dpi=1200, constrained_layout=True)

# read, modify, and plot original (OG) image
imgOG = iio.imread('B6_DAPI_1.tif')
sizeY = imgOG.shape[0]; sizeX = imgOG.shape[1]
pixel_count = sizeX * sizeY
# convert to grayscale, if needed, to simplify analysis
if len(imgOG.shape) == 3:
    imgOG = ski.color.rgb2gray(imgOG)
# reduce size of image, if needed, to fit max pixels defined
if REDUCE == True:
    while pixel_count > MAX_PIXELS:
        print("Too many pixels! Reducing from ", pixel_count)
        imgOG = scipy.zoom(imgOG, 0.75)
        sizeY = imgOG.shape[0]; sizeX = imgOG.shape[1]
        pixel_count = sizeX * sizeY
OG = Image("Original Image", imgOG, PLOC1)
max_val = np.max(OG.data)
min_val = np.min(OG.data)
#print(f"Max/min of OG img = {max_val}/{min_val}, data-type = {type(OG.data[10][10])}")
plot_info = OG.easy_plot_info()
easy_plot(-plot_info[0], plot_info[1], plot_info[2])

# basic empty frame, same size as OG img, used for setting up other images
frame = np.zeros((sizeY, sizeX))

# image preprocessing
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # adaptive contrast adjust
OG.data = clahe.apply(OG.data)
plot_info = OG.easy_plot_info()
easy_plot(-plot_info[0], PLOC4, "After Preprocessing")

# apply LoG filter with varying σ's to find different sized blobs
sigma = [1, 2, 3, 4]
Log0 = Image("Empty", frame, (0,0))
Log1 = Image(f"LoG-ified σ = {sigma[0]}", frame, PLOC2)
Log2 = Image(f"LoG-ified σ = {sigma[1]}", frame, PLOC3)
Log3 = Image(f"LoG-ified σ = {sigma[2]}", frame, PLOC5)
Log4 = Image(f"LoG-ified σ = {sigma[3]}", frame, PLOC6)
Logs = [Log0, Log1, Log2, Log3, Log4]
for n in range(1, len(Logs)):
    # ensure that LoG can yield negative values by changing to signed data
    Logs[n].data = scipy.gaussian_laplace((OG.data).astype(np.int16), sigma[n-1])
    max_val = np.max(Logs[n].data)
    min_val = np.min(Logs[n].data)
    #print(f"Max/min of LoG{n} = {max_val}/{min_val}")
    plot_info = Logs[n].easy_plot_info()
    easy_plot(plot_info[0], plot_info[1], plot_info[2], colors='tab10')

# nucleus markers for LoGs
for n in range(1, len(Logs)):
    mag_data = np.sqrt((Logs[n].data) ** 2)
    maxima = ski.feature.peak_local_max(mag_data, min_distance=20,
                                        exclude_border=0, threshold_abs=20)
    easy_plot(maxima, Logs[n].ploc, plotType='pointOverlay', colors='ks')
    print(f"Nuclei count {n}: ", int(maxima.size/2))