import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from skimage import measure, filters, color
from scipy import ndimage



# ~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~
def rgb_to_gscale(img):
# take the data from a 3-dimensional RGB image [height][width][RGB],
# convert each pixel to gray via row x row and column x column operations,
# return an image as an array of data, the same size as the input image,
# with the RGB values of each pixel averaged to the same value.
# e.g. input pixel [123][249][184] -> output pixel [204][204][204].
#                  seafoam green                   neutral grey """
    print("Computing grayscale image.")
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


def display_BW_histogram(hgram, y, x):

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
    
    normalizedData[0] = 0 #   <<< remove pure blacks for more informative graphs
    axarr[y][x].bar(xAxis, normalizedData, )
    axarr[y][x].set_xticks([0, 51, 102, 153, 204, 255])
    #print("Area under the histogram = ", normPixelCount)
    
    return 0


def binarize_gscale(img):
    
    data = img
    threshold = filters.threshold_otsu(data)
    #print("Computing binary black and white image with threshold = ", threshold)
    bin_img = (data > threshold)
    bin_img = ndimage.binary_fill_holes(bin_img)
    
    return bin_img


def label_and_crop_last(imgBW, imgOriginal, invertParam):
    
    data = imgBW
    originalData = imgOriginal
    dataLabels = measure.label(imgWinter_binBW, connectivity=2)
    #groupCount = dataLabels.max()
    #coloredImg = color.label2rgb(dataLabels)
    height = data.shape[0]
    width = data.shape[1]
    data_out = np.zeros((height, width))
    #print("Number of objects identified = ", groupCount)
    targetLabel = dataLabels[500][500] # <<< rough pixel location of the desired object
    
    h = 0
    w = 0
    for h in range(height):
        for w in range(width):
            if invertParam != 1: #   <<< if invertParam is true, then desired object is cropped instead of background.
                if dataLabels[h][w] == targetLabel: #   <<< desired object label
                    data_out[h][w] = originalData[h][w]
                else:
                    data_out[h][w] = 0
            else:
                if dataLabels[h][w] != targetLabel: #   <<< desired object label
                    data_out[h][w] = originalData[h][w]
                else:
                    data_out[h][w] = 0
                
    return data_out


def hgramPixelCount(hgram):
    
    data = hgram
    pixels = 0
    width = len(data)
    for w in range(width):
        pixels = pixels + data[w]
        
    return pixels
    

def overlay_histogram(hgram, y, x, z): # y and x in this case being the axis of the original plot being overlayed, and z being the number of pixels of the original hgram.
    
    data = hgram
    width = len(data)
    xAxis = np.arange(0, width, 1)
    
    # normalize so that area under the graph
    pixelCount = 0
    for i in range(1, width): # count the number of pixels tallied from the input histogram data.
        pixelCount = pixelCount + data[i]
    # pixel count now equals some large number (e.g. 560,000) equal to h x w of image.
    # need to convert the data so that each pixel is a float, and all the float pixels sum to 1.
    normFactor = 1/(pixelCount/(pixelCount/z))
    data = data.astype(float)
    normalizedData = np.zeros(width, dtype=float)
    normPixelCount = 0
    for i in range(width):
        normalizedData[i] = data[i] * normFactor
        normPixelCount = normPixelCount + normalizedData[i]
    
    normalizedData[0] = 0 #   <<< remove pure blacks for more informative graphs
    overlay = axarr[y][x].twinx()
    overlay.bar(xAxis, normalizedData, color='red')
    overlay.sharey(axarr[y][x])
    
    return 0

    
# ~~~~~~~~~~ MAIN ~~~~~~~~~~
# setup for plotting multiple images
f, axarr = plt.subplots(5,2)
f.tight_layout()

imgWinter = iio.imread("Winter.png") # 1.a. read and store image file.
axarr[0][0].imshow(imgWinter) # 1.a. display in color (top left plot).
axarr[0][0].set_title('Original Image')

imgWinter_gscale = rgb_to_gscale(imgWinter) # convert original image to grayscale and copy onto new array.
axarr[0][1].imshow(imgWinter_gscale, cmap='gray') # 1.a. display in grayscale (bottom left plot).
axarr[0][1].set_title('Grayscale')

imgWinter_BW_hgram = compute_BW_histogram(imgWinter_gscale) # count each pixel and sort into intensity values.
display_BW_histogram(imgWinter_BW_hgram, 1, 1) # 1.b. plot the black and white histogram normalized so that area under curve = 1 (bottom right plot).
axarr[1][1].set_title('Grayscale Histogram')

# 1.c. crop any region of 100 x 100 pixels where there is snow on the ground.
#   methodology:
#   (1) binarize image (black XOR white, no gray)
#   (2) identify different sections (e.g. sky, trees+building+mountains, and ground)
#   (3) remove snow on ground (the white pixels contained in the ground section)
imgWinter_binBW = binarize_gscale(imgWinter_gscale)
#axarr[2][0].imshow(imgWinter_binBW, cmap='gray')
imgWinterBackground = label_and_crop_last(imgWinter_binBW, imgWinter_gscale, 1) # 1.c. crop any region of 100 x 100 pixels where there is snow on the ground.
axarr[2][0].imshow(imgWinterBackground, cmap='gray') # 1.c. display.
axarr[2][0].set_title('Snow cropped')

imgWinterSnow = label_and_crop_last(imgWinter_binBW, imgWinter_gscale, 0)
axarr[2][1].imshow(imgWinterSnow, cmap='gray')
axarr[2][1].set_title('Background cropped')

winterBackground_hgram = compute_BW_histogram(imgWinterBackground)
display_BW_histogram(winterBackground_hgram, 3, 0)
axarr[3][0].set_title('Background Histogram')
snow_hgram = compute_BW_histogram(imgWinterSnow)
display_BW_histogram(snow_hgram, 3, 1)
axarr[3][1].set_title('Snow Histogram')

# 1.d. compare the histograms of the entire image and the subregion by overlaying on top of eachother.
display_BW_histogram(imgWinter_BW_hgram, 1, 0)
axarr[1][0].set_title("Overlayed Histograms")
winterPixelCount = hgramPixelCount(imgWinter_BW_hgram)
overlay_histogram(snow_hgram, 1, 0, winterPixelCount)

# 1.e. read Desert.png.
imgDesert = iio.imread("Desert.png")
imgDesert_gscale = rgb_to_gscale(imgDesert) # 1.e. convert to BW image.
axarr[4][0].imshow(imgDesert_gscale, cmap='gray') # 1.e. display
axarr[4][0].set_title('Desert Grayscale')

# 1.f. compare the histograms of (e) and (b) by overlaying them.
desert_hgram = compute_BW_histogram(imgDesert_gscale)
display_BW_histogram(desert_hgram, 4, 1)
axarr[4][1].set_title('Desert and Winter')
desertPixelCount = hgramPixelCount(desert_hgram)
overlay_histogram(imgWinter_BW_hgram, 4, 1, desertPixelCount)

# 1.f. can these 2 images be discriminated by their histograms?
print("Based on the histograms alone, the desert and winter scenes can identified.")
print("The winter scene (blue) has a significant # of pixels on the white-colored side,")
print("while the desert (blue) has a more even distribution towards the neutral-colors,")
print("just as we would expect to see looking at a desert and winter scene in real life.")