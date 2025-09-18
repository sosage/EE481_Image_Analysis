# ~~~~~~~~~~ LIBRARIES ~~~~~~~~~~

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from skimage import exposure
#from scipy import ndimage



# ~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~

# take an RGB image and convert to grayscale.
def convert_rgb_gscale(img):
    
    print("Generating grayscale image.")

    data = img
    height = data.shape[0]
    width = data.shape[1]
    data_out = np.zeros((height, width))

    h = 0
    w = 0
    for h in range(height):
        for w in range(width):
            red_ch = data[h][w][0]
            grn_ch = data[h][w][1]
            blu_ch = data[h][w][2]
            gray_avg = red_ch//3.34 + grn_ch//1.7 + blu_ch//8.77 # RGB to gray using luminosity method.
            data_out[h][w] = gray_avg

    return data_out



def compute_BW_histogram(img):
# take the data from a 2-dimensional grayscale image [height][width],
# scan along each pixel row x row and column x column, keeping count
# of which pixels are closest to which intensity value (i.e. 255 possible values,
# 0 =  least intense, and 255 = most intense), and return an array of [255] values
# with each value being the number of pixels that matched that intensity in the image. """

    print("Computing histogram.")
    data = img.astype(int)
    height = data.shape[0]
    width = data.shape[1]
    data_out = np.zeros(256) # array to contain the pixel-intensity tally of the input image.
    
    h = 0
    w = 0
    value = 0
    for h in range(height):
        for w in range(width):
            value = data[h][w] # check value of pixel and increment tally of that value by 1.
            data_out[value] = data_out[value] + 1

    return data_out



def display_BW_histogram(hgram, plotObj, y, x):

    data = hgram
    width = len(data)
    xAxis = np.arange(0, width, 1)

    # normalize so that area under the graph = set value.
    totalArea = 1 #   <<< set value here
    pixelCount = 0
    for i in range(width): # count the number of pixels tallied from the input histogram data.
        pixelCount = pixelCount + data[i]
    #print("Total pixel count = ", pixelCount)
    # pixel count now equals some large number (e.g. 560,000) equal to h x w of image.
    # need to convert the data so that each pixel is a float, and all the float pixels sum to 1.
    normFactor = 1/(pixelCount/totalArea)
    data = data.astype(float)
    normalizedData = np.zeros(width, dtype=float)
    normPixelCount = 0
    #maxNormDataPoint = 0
    for i in range(width):
        normalizedData[i] = data[i] * normFactor
        normPixelCount = normPixelCount + normalizedData[i]
    
    #normalizedData[0] = 0 #   <<< remove pure blacks for more informative graphs
    plotObj[y][x].bar(xAxis, normalizedData)
    plotObj[y][x].set_xticks([0, 51, 102, 153, 204, 255])
    #print("Area under the histogram = ", normPixelCount)
    
    return 0



# enhance image using histogram equalization
def hgram_eq_enhance(img):
    
    data = img
    enhancedImg = exposure.equalize_hist(data)
    
    return enhancedImg

# ~~~~~~~~~~ MAIN CODE ~~~~~~~~~~

# task 2: use lenna image to 
# (i.) convert color to gray scale
# (ii.) downsample the image by 3 levels to
# build a pyramid representation [comments on the results].

imageSource = 'valley.tif'
imgOriginal = iio.imread(imageSource)

# gscale
imgGray = convert_rgb_gscale(imgOriginal)

# enhance and compute histograms
hgramGray = compute_BW_histogram(imgGray)
imgEnhanced = hgram_eq_enhance(imgGray)
hgramEnhanced = np.histogram(imgEnhanced, bins=range(257))
print("Shape of hgram: ", hgramEnhanced.shape)

# plot images
plt.rc('font', size=8)
f, ax = plt.subplots(2, 2, dpi=1000, constrained_layout=True)

ax[0][0].imshow(imgOriginal)
ax[0][0].set_title('Original Image')
ax[1][0].imshow(imgEnhanced, cmap='gray')
ax[1][0].set_title('Enhanced Image')
display_BW_histogram(hgramGray, ax, 0, 1)
ax[0][1].set_title('Original Histogram')
ax[1][1].hist(hgramEnhanced, bins=257, orientation='horizontal')
ax[1][1].set_title('Enhanced Histogram')

