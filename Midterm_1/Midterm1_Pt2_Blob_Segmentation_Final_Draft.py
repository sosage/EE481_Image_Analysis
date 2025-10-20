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
                            markersize=2)
    if gridshare == 1:
        ax[rowI][colI].sharex(ax[0][0])
        ax[rowI][colI].sharey(ax[0][0])
    
    return 0;

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

# flood fill
def flood_fill(imgIn):
    datatype_in = type(imgIn)
    # Invert the image: black objects become white, white holes become black
    inverted_for_black_holes = cv2.bitwise_not(imgIn)

    # Now, fill the "black holes" in this inverted image (which were white holes in the original)
    # Using dilation on the inverted image will fill these black holes
    kernel = np.ones((5, 5), np.uint8)
    filled_inverted = cv2.dilate(inverted_for_black_holes, kernel, iterations=1)

    # Invert back to get the original black objects with filled white holes
    filled_black_sections = cv2.bitwise_not(filled_inverted)
    
    return filled_black_sections.astype(datatype_in)


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
plot_info = OG.easy_plot_info()
easy_plot(-plot_info[0], plot_info[1], plot_info[2])

# basic empty frame, same size as OG img, used for setting up other images
frame = np.zeros((sizeY, sizeX))

# image preprocessing
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # adaptive contrast adjust
OG.data = clahe.apply(OG.data)
plot_info = OG.easy_plot_info()
easy_plot(-plot_info[0], PLOC4, "After Preprocessing")

# LoG filter to find nuclei
sigma = 3
Log1 = Image(f"LoG-ified σ = {sigma}", frame, PLOC2)
Log1.data = scipy.gaussian_laplace((OG.data).astype(np.int16), sigma)
max_val = np.max(Log1.data)
min_val = np.min(Log1.data)
plot_info = Log1.easy_plot_info()
easy_plot(plot_info[0], plot_info[1], plot_info[2], colors='tab10')

# nucleus markers for LoGs
mag_data = np.sqrt((Log1.data) ** 2)
maxima = ski.feature.peak_local_max(mag_data, min_distance=20,
                                    exclude_border=0, threshold_abs=20)
print("Nuclei count: ", int(maxima.size/2))

# Z-cross to find edges of cells
ZX = Image("Zero X'ings", frame, PLOC5)
ZX.data = calc_zcross(Log1.data)
plot_info = ZX.easy_plot_info()
easy_plot(plot_info[0], plot_info[1], plot_info[2])
# process for analysis
ZXedit = Image("ZX'ings Postprocessing", frame, PLOC3)
ZXedit.data = (ZX.data).astype(np.int16)*255
ZXedit.data = cv2.GaussianBlur(ZXedit.data, (21,21), 0) # blurring
ZXedit.data = flood_fill(ZXedit.data) # fill white holes in black sections
thresh = ski.filters.threshold_otsu(ZXedit.data) # thresholding
ZXedit.data = ZXedit.data > thresh
plot_info = ZXedit.easy_plot_info()
easy_plot(plot_info[0], plot_info[1], plot_info[2])

# watershed segmentation
topography = np.zeros((ZXedit.data).shape, dtype=bool) # convert to mask image
topography[tuple(maxima.T)] = True # assign coordinates in mask image = true
labels, count = scipy.label(topography)
Segments = Image("Segments", frame, PLOC6)
Segments.data = ski.segmentation.watershed(topography, labels, mask=ZXedit.data)
plot_info = Segments.easy_plot_info()
easy_plot(plot_info[0], plot_info[1], plot_info[2], colors='nipy_spectral_r')
easy_plot(maxima, Segments.ploc, plotType='pointOverlay', colors='w*')