# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 14:26:32 2025
@author: S.W. Marsden
"""

import numpy as np
import skimage as ski
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# ~~~~~~~~~ DEFINITIONS ~~~~~~~~~
PLOT_GRID = (2, 4)
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

import pt2_funcs as skyler
# homemade function library. contains:
# expand_canvas(img, extra_pixels, value=0)
# assign_point_curves(img, init_direction=('R','D'))
# compute_curvature(pointcurve_data)
# create_curvature_img(curvatures, pointcurve_map, base_img)

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
                           dpi=800, constrained_layout=True)
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
                            markerfacecolor='w')
    
    if gridshare == 1:
        ax[rowI][colI].sharex(ax[0][0])
        ax[rowI][colI].sharey(ax[0][0])
   
    return 0;

# take a square section from top left of an image for detail analysis
def snip(img, sizeX=300, sizeY=300):
    img_snip = img[:sizeY,:sizeX]
    return img_snip;

# ~~~~~~~~~~ MAIN CODE ~~~~~~~~~~
ax = plotArray_init()
Beans = iio.imread('CoffeeBeans.tif')

# convert to grayscale, if needed, to simplify analysis
grayBeans = Grayout(Beans)
# blur and threshold to create smooth binary mask
smoothBeans = ski.filters.gaussian(grayBeans, sigma=0.5)
binaryBeans = smoothBeans < (ski.filters.threshold_otsu(smoothBeans))

# distance transform and watershed segment
distance = ndi.distance_transform_edt(binaryBeans)
coords = peak_local_max(distance, min_distance=9, exclude_border=0)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, bean_count = ndi.label(mask)
labeledBeans = watershed(-distance, markers, mask=binaryBeans)
bean_properties = regionprops(labeledBeans)

# find biggest bean and put into its own image
biggestBean = (0, 0) # placeholder, i=0 is label and i=1 is size in pixels
for i in bean_properties:
    #print(f"#{i.label} has size of {i.area} at {i.centroid}")
    if i.area > biggestBean[1]:
        biggestBean = (i.label, i.area)
print(f"Biggest bean award: #{biggestBean[0]} at {biggestBean[1]} pixels!")
bigBean = bean_properties[biggestBean[0]-1].image
bigBean = skyler.expand_canvas(bigBean, 5)
RGBean = (ski.color.gray2rgb(bigBean*255)).astype(np.uint8)
iio.imwrite('bigBean', RGBean, extension='.png')
edgeBean = ski.filters.roberts(bigBean) # find edges, make 1-pixel thick
edgeBean[edgeBean != 0] = 1 # ensure values are binary for pointcurve func
pointcurve_bigbean = skyler.assign_point_curves(edgeBean)
curvature_bigbean = skyler.compute_curvature(pointcurve_bigbean)
curveimg = skyler.create_curvature_img(curvature_bigbean,
                                       pointcurve_bigbean,
                                       edgeBean)
curve_and_bean = curveimg + edgeBean

"""bigBean = iio.imread('rotated_bigbean.png')
bigBean = ski.filters.gaussian(bigBean, sigma=1)
bigBean = Grayout(bigBean)
bigBean = bigBean > (ski.filters.threshold_otsu(bigBean))"""

"""
bgSize = 80
background = np.zeros((bgSize,bgSize), dtype=int)
bigBeanPixels = []
for r in range(len(bigBean)):
    for c in range(len(bigBean[0])):
        if bigBean[r][c] == True:
            bigBeanPixels.append((r+(0.2*bgSize), c+(0.25*bgSize)))
bigBeanPixels = np.array(bigBeanPixels, dtype=int)
for i in bigBeanPixels:
    background[i[0],i[1]] = True
bigBean = background
"""
        
"""
# compute curvature and plot
edgeBean = ski.filters.roberts(bigBean)
for r in range(bgSize):
    for c in range(bgSize):
        if edgeBean[r][c] != 0:
            edgeBean[r][c] = 1
"""

# create a list of image objects for plotting
IMG = []
IMG.append(Image("CoffeeBeans.tif", Beans, PLOC[0])) #IMG[0]
IMG.append(Image("thresholded", binaryBeans, PLOC[1])) #IMG[1]
IMG.append(Image("distance trans", distance, PLOC[2])) #IMG[2]
IMG.append(Image("segmented", labeledBeans, PLOC[3])) #IMG[3]
IMG.append(Image(f"biggest bean, #{biggestBean[0]}", bigBean, PLOC[4])) #IMG[4]
IMG.append(Image("edge", edgeBean, PLOC[5])) #IMG[5]
IMG.append(Image("curvature", curveimg, PLOC[6])) #IMG[6]
IMG.append(Image("curvature + edges", curve_and_bean, PLOC[7])) #IMG[7]
# plot all images in the IMG list
# plot the segmented image
ezplot(IMG[3].ezplot_info(), ax, colors=plt.cm.nipy_spectral_r)
# display segmented beans labels
bean_coords = []
for i in bean_properties:
    bean_coords.append(np.array(i.centroid, dtype=int))
bean_coords = np.array(bean_coords)
ezplot( (bean_coords, PLOC[3], "points"), ax, plotType='pointOverlay', colors='w.')
ax[PLOC[3][0]-1][PLOC[3][1]-1].set_axis_off()
for n in range(bean_coords.shape[0]):
    ax[PLOC[3,0]-1][PLOC[3,1]-1].annotate(n+1, xy=(bean_coords[n][1], bean_coords[n][0]),
                                          textcoords='data', horizontalalignment='left',
                                          verticalalignment='top', color='black')
# plot the curvature image
ezplot(IMG[6].ezplot_info(), ax, colors=plt.cm.nipy_spectral)
# plot the curvature superimposed onto the edge image
ezplot(IMG[7].ezplot_info(), ax, colors=plt.cm.nipy_spectral)
# plot everything not done yet
dont_plot = [3, 6, 7]
for i in IMG:
    if not(IMG.index(i) in dont_plot):
        ezplot(i.ezplot_info(), ax)
