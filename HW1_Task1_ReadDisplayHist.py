import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

# ~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~
def rgb_to_gscale(img):
# takes a 3-dimnesional RGB image array [height][width][RGB],
# convert each pixel to gray via row x row and column x column operations,
# returning an image as an array of data, the same size as the input image,
# with the RGB values of each pixel averaged to the same value.
    
    data = img
    height = data.shape[0]
    width = data.shape[1]
    channels = data.shape[2]
    data_out = data.copy() # make a copy of the input image to be written on for output.
    
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


def compute_histogram(img):
    ............

# ~~~~~~~~~~ MAIN ~~~~~~~~~~
plt.figure() # setup for plotting multiple images
f, axarr = plt.subplots(2,1)

image_data = iio.imread("Winter.png") # read and store image file.
axarr[0].imshow(image_data) # 1.a. display in color

gscale_image = rgb_to_gscale(image_data) # convert original image to grayscale and copy onto new array.
axarr[1].imshow(gscale_image) # 1.b. display in grayscale
print(gscale_image[110][100])