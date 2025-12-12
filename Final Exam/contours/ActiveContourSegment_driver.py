"""
Image Analysis - Final Exam - Question 2 - Active Contour
Segmentation Driver File
Created on Thu Dec  11 19:15:00 2025

@author: S.W. Marsden
"""

#import numpy as np
#from numpy.lib.stride_tricks import sliding_window_view
#import matplotlib.pyplot as plt

import imageio.v3 as iio

import ActiveContourSegment_funcs as skyler

class Image:
    def __init__(self, name, data, ploc):
        self.name = name
        self.data = data
        self.ploc = ploc
        
    def ezplot_info(self):
        return self.data, self.ploc, self.name;

# global variables
plot_rows = 4
plot_cols = 4
smoothing_iters = 12

# setup the multiple plots for output visualization.
plotGrid = (plot_rows, plot_cols)
PLOC = skyler.initialize_plot_locations(plotGrid)
ax = skyler.initialize_multiplot(plotGrid, 
                                 text_size=8,
                                 res=800)

# read in the initial images, convert to grayscale,
# apply total variation smoothing, and
# add to Image list.
IMG = []
img1 = iio.imread('Cryo_EM_Image1.jpeg')
img2 = iio.imread('Cryo_EM_Image2.png')

IMG.append(Image('cryo 1', # img [0]
                 skyler.convert_to_gray(img1),
                 PLOC[0]))

IMG.append(Image('tot var smooth, C1', # img [1]
                 skyler.total_var_smooth(
                     IMG[0].data,
                     intensity=10,
                     loops=smoothing_iters*3),
                 PLOC[1]))

IMG.append(Image('cryo 2', # img [2]
                 skyler.convert_to_gray(img2), PLOC[2]))

IMG.append(Image('tot var smooth, C2', # img [3]
                 skyler.total_var_smooth(
                     IMG[2].data,
                     loops=smoothing_iters/2),
                 PLOC[3]));

# find the chan-vese segmentation for each image over 40
# iterations, in batches of 10 iterations each.
for i in range(0,4):
    for n in range(0, 4):
        iterations = 10+n*10
        segments = skyler.chan_vese_segment(IMG[i].data,
                                            iters=iterations)
        IMG.append(Image(f"C{i+1}, i={iterations}",
                         segments, PLOC[n+(i*4)]))
        skyler.ezplot(IMG[4+(i*4)+n].ezplot_info(), ax)
# ~~~~~~~~~~plot image guide~~~~~~~~~~
#                 i=10  i=20  i=30  i=40
# CRYO1_original: I[ 0] I[ 1] I[ 2] I[ 3]
# CRYO1_smoothed: I[ 4] I[ 5] I[ 6] I[ 7]
# CRYO2_original: I[ 8] I[ 9] I[10] I[11]
# CRYO2_smoothed: I[12] I[13] I[14] I[15]

# plot all images not in the "dont plot" list of IMG idxs.
dont_plot = []
for i in IMG:
    if not(IMG.index(i) in dont_plot):
        skyler.ezplot(i.ezplot_info(), ax)
        
        
"""snakeC1 = skyler.find_contour(IMG[2].data)
skyler.ezplot((snakeC1,PLOC[3], 'C1 contour'), ax, plotType='contour')
"""

"""test_img = skyler.total_var_smooth(IMG[1].data, loops=smoothing_iters)
IMG.append(Image('test img', # img [-1]
                 test_img, PLOC[-1]))
"""
