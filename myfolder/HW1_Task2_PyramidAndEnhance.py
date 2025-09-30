# ~~~~~~~~~~ LIBRARIES ~~~~~~~~~~

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
#from skimage import measure, filters, color
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


# take a gscale image and downscale using sliding window operations.
def downscale(img, a):
    
    print("Generating downscaled image.")
    
    data = img
    window_parameters = (a, a) #    <<< paramter a = size of one side of the box
    windowArray = np.lib.stride_tricks.sliding_window_view(data, window_parameters)[::a, ::a]
    # ^ 2x2 array of windows containing 2x2 slices of the image  ^ window size        ^ step sizes
    height = windowArray.shape[0]
    width = windowArray.shape[1]
    downscaledImg = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            
            section = windowArray[y][x]
            avgValue = np.sum(section * (1 / (a * a)))
            downscaledImg[y][x] = avgValue
                    
    """
    
    Methodology:
    [* *]       [ 1/4  1/4 ]       [a b]
    [* *]   *   [ 1/4  1/4 ]   =   [c d]   --->   a+b+c+d = downscaled pixel   --->   slide window over, repeat
    
    image       kernel
    window
    
    """
    
    return downscaledImg
    
# ~~~~~~~~~~ MAIN CODE ~~~~~~~~~~

# task 2: use lenna image to 
# (i.) convert color to gray scale
# (ii.) downsample the image by 3 levels to
# build a pyramid representation [comments on the results].

imageSource = 'Lenna.png'
imgLenna = iio.imread(imageSource)

# gscale
imgLennaGray = convert_rgb_gscale(imgLenna)
print("Grayscale image size: ", imgLennaGray.shape)

# level 1 downscale
imgLennaHalfSize = downscale(imgLennaGray, 2)
print("Downsize 1 image size: ", imgLennaHalfSize.shape)

# level 2 downscale
imgLennaQuarterSize = downscale(imgLennaHalfSize, 2)
print("Downsize 2 image size: ", imgLennaQuarterSize.shape)

# level 3 downscale
imgLennaEigthSize = downscale(imgLennaQuarterSize, 2)
print("Downsize 3 image size: ", imgLennaEigthSize.shape)

# plot images
plt.rc('font', size=8)
f, ax = plt.subplots(2, 2, dpi=1000, constrained_layout=True)
#f.tight_layout()

#ax[0][0].imshow(imgLenna)
#ax[0][0].set_title('Original Image')
ax[0][0].imshow(imgLennaGray, cmap='gray')
ax[0][0].set_title('Grayscale, 512x512')
ax[0][1].imshow(imgLennaHalfSize, cmap='gray')
ax[0][1].set_title('Downsized, 256x256')
ax[1][0].imshow(imgLennaQuarterSize, cmap='gray')
ax[1][0].set_title('Downsized, 128x128')
ax[1][1].imshow(imgLennaEigthSize, cmap='gray')
ax[1][1].set_title('Downsized, 64x64')