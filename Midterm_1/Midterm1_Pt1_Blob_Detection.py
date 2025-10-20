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
                            markersize=3, markerfacecolor='none')
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

# Compute zero crossings of an image.
def calc_zcross(imgIn):

    imgLap = scipy.laplace(imgIn)
    imgLap_signed = np.sign(imgLap)

    # Pad the signed_laplacian to handle boundary conditions during difference calculation
    # This ensures we can compare each pixel with its neighbors
    imgLSP = np.pad(imgLap_signed, ((0, 1), (0, 1)), mode='edge')

    # Check for sign changes in horizontal and vertical directions
    # A zero crossing occurs if a pixel and its neighbor have different signs
    diff_x = imgLSP[:-1, :-1] - imgLSP[:-1, 1:]
    diff_y = imgLSP[:-1, :-1] - imgLSP[1:, :-1]

    # Zero crossings are where the absolute difference is not zero (i.e., signs are different)
    zero_crossings = np.logical_or(np.abs(diff_x) > 0, np.abs(diff_y) > 0)

    return zero_crossings.astype(float) # Convert boolean result to float for visualization


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
print(f"Max/min of OG img = {max_val}/{min_val}, data-type = {type(OG.data[10][10])}")
plot_info = OG.easy_plot_info()
easy_plot(plot_info[0], plot_info[1], plot_info[2])

# basic empty frame, same size as OG img, used for setting up other images
frame = np.zeros((sizeY, sizeX))

# image preprocessing
#OG.data = cv2.GaussianBlur(OG.data, (5,5), 0) # smoothing
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # adaptive contrast adjust
OG.data = clahe.apply(OG.data)
#thresh = ski.filters.threshold_otsu(OG.data) # binarization
#OG.data = OG.data > thresh
#OG.data = ski.exposure.equalize_hist(OG.data) # basic contrast adjust
plot_info = OG.easy_plot_info()
easy_plot(plot_info[0], PLOC4, "After Preprocessing")

# blob detection loop, try to yield blob count = manual count.
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
    print(f"Max/min of LoG{n} = {max_val}/{min_val}")
    plot_info = Logs[n].easy_plot_info()
    easy_plot(plot_info[0], plot_info[1], plot_info[2])

"""
# find the zero crossings of the LoG-ified image
ZX0 = Image("Empty", frame, (0,0))
ZX1 = Image("Zero X-ings", frame, PLOC5)
ZX2 = Image("Zero X-ings", frame, PLOC8)
ZX3 = Image("Zero X-ings", frame, PLOC11)
ZXs = [ZX0, ZX1, ZX2, ZX3]
for n in range(1, len(ZXs)):
    ZXs[n].data = calc_zcross(Logs[n].data)
    plot_info = ZXs[n].easy_plot_info()
    easy_plot(plot_info[0], plot_info[1], plot_info[2])
"""

# nucleus markers for LoGs
for n in range(1, len(Logs)):
    mag_data = np.sqrt((Logs[n].data) ** 2)
    maxima = ski.feature.peak_local_max(mag_data, min_distance=25,
                                        exclude_border=0, threshold_abs=20)
    easy_plot(maxima, Logs[n].ploc, plotType='pointOverlay', colors='bs')
    print(f"Nuclei count {n}: ", int(maxima.size/2))
    
"""
# find the nucleus marker
print(Logs[1].data[96][100])
#Log1.data = np.sqrt(Log1.data ** 2) # turn minimums into maximums
hiPoints = ski.feature.peak_local_max(Log1.data, min_distance=55, exclude_border=0)
easy_plot(hiPoints, Log1.ploc, plotType='pointOverlay', colors='r.')
print("Coffee bean count 1: ", int(hiPoints.size/2))
hiPoints = ski.feature.peak_local_max(Log2.data, min_distance=55, exclude_border=0)
easy_plot(hiPoints, Log2.ploc, plotType='pointOverlay', colors='r.')
print("Coffee bean count 2: ", int(hiPoints.size/2))
hiPoints = ski.feature.peak_local_max(Log3.data, min_distance=15, exclude_border=0)
easy_plot(hiPoints, Log3.ploc, plotType='pointOverlay', colors='r.')
print("Coffee bean count 3: ", int(hiPoints.size/2))
"""

# fix plot spacing
#plt.tight_layout(pad=0, w_pad=0, h_pad=0)

"""
# apply LoG filter with varying sigmas to find different sized blobs
Log = Image("LoG-ified", frame, PLOC4)
sigma = 2
Log.data = scipy.gaussian_laplace(OG.data, sigma)
plot_info = Log.easy_plot_info()
easy_plot(plot_info[0], plot_info[1], plot_info[2])

# find the zero crossings of the LoG-ified image
ZX = Image("Zero X-ings", frame, PLOC5)
ZX.data = calc_zcross(Log.data)
plot_info = ZX.easy_plot_info()
easy_plot(plot_info[0], plot_info[1], plot_info[2])
"""

"""
# find the markers (low point of the basins)
lowPoints = np.argwhere(ski.morphology.local_minima(Log.data))
#lowPoints = ski.feature.peak_local_max(imgDistTrans, footprint=np.ones((5, 5)))
easy_plot(lowPoints, PLOC3, plotType='pointOverlay', colors='r.')
"""
