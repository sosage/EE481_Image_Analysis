"""
Created on Sat Nov 29 17:37:30 2025
HOMEWORK 5: COFFEE BEAN PT 2 PROCESSING FUNCTIONS
@author: S.W. Marsden
"""

import numpy as np
import skimage as ski
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import convolve2d

def expand_canvas(img, extra_pixels, value=0):
    img1 = np.array(img)
    size_y1 = img1.shape[0]
    size_x1 = img1.shape[1]
    size_y2 = size_y1 + 2*extra_pixels
    size_x2 = size_x1 + 2*extra_pixels
    img2 = np.full((size_y2, size_x2), value)
    for row in range(extra_pixels, size_y1 + extra_pixels):
        for col in range(extra_pixels, size_x1 + extra_pixels):
            grid = (row - extra_pixels, col - extra_pixels)
            img2[row,col] = img1[grid]
    #print(f"row/col 1: {size_y1}/{size_x1}. row/col 1: {size_y2}/{size_x2}.")
    return img2;

def compute_derivatives(img, sigma):
    # a = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1)
    # e.g. k = 9 and s = 2 yields kernel = [-4 -3 -2 1 0 +1 +2 +3 +4]
    """dx = gaussian_filter(img, sigma, order=(1,0))
    dy = gaussian_filter(img, sigma, order=(0,1))
    d2x = gaussian_filter(img, sigma, order=(2,0))
    d2y = gaussian_filter(img, sigma, order=(0,2))"""
    dx = gaussian_filter1d(img, sigma, order=1, radius=9, axis=1, mode='constant', cval=0)
    dy = gaussian_filter1d(img, sigma, order=1, radius=9, axis=0, mode='constant', cval=0)
    d2x = gaussian_filter1d(img, sigma, order=2, radius=9, axis=1, mode='constant', cval=0)
    d2y = gaussian_filter1d(img, sigma, order=2, radius=9, axis=0, mode='constant', cval=0)
    #(data, sigma=2, mode='constant', cval=0)
    
    return dx, dy, d2x, d2y;

def compute_curvature(img, x1, y1, x2, y2):
    # compute curvature in accordance with formula for K
    #           x' y" - y' x"
    # K = --------------------------
    #      [ (x')^2 + (y')^2 ]^(3/2)
    """
    dx = x1
    dy = y1
    d2x = x2
    d2y = y2
    top = (dx * d2y) - (dy * d2x)
    bottom = ((dx**2) + (dy**2)) ** (3/2)
    K = top / bottom
    """
    rows = len(img)
    cols = img[0].size
    K = np.zeros((rows,cols))
    for row in range(0, rows):
        for col in range(0, cols):
            dx = x1[row][col]
            dy = y1[row][col]
            d2x = x2[row][col]
            d2y = y2[row][col]
            top = ((dx * d2y) - (dy * d2x))
            bottom = ((dx**2) + (dy**2)) ** (1.5)
            if bottom != 0:
                K[row][col] = (top / bottom)
            if abs(K[row][col]) > 0.01:
                K[row][col] = 0
            if row == 136 and col == 34:
                print("dx, dy, d2x, d2y, K for point:[136,34]")
                print(dx, dy, d2x, d2y, K[136][34])
                
    
    #print("dx, dy, d2x, d2y, K for point:[136,34]")
    #print(dx[136][34], dy[136][34], d2x[136][34], d2y[136][34], K[136][34])
    return K;

# take a 1-pixel thick curve and assign indexes to each point along the curve.
# img input must be 1-pixel thick (roberts edge) and binary (curve=1 bgnd=0).
# initial direction parameter describes how to follow the curve...
# ... e.g. ('R','D') attempts to find a point to the right first, down second,
# left third, and up fourth. These must be in the order L/R then U/D.
def assign_point_curves(img, init_direction=('R','D')):
    point_num = np.sum(img != 0)
    point = []
    for i in range(point_num):
        point.append((-1,-1))
    point = np.array(point)
    print(f"Number of points along curve: {point_num}")
    
    # find the initial point to start the curve at.
    found = False
    size = (img.shape[0],img.shape[1])
    for row in range(0, size[0]):
        if found == False:
            for col in range(0, size[1]):
                if img[row,col] != 0:
                    point[0] = (row, col)
                    found = True
                    break
    
    # find the other points while following the curve all the way around.
    d1 = init_direction[0] # direction values control direction sequence
    d2 = init_direction[1]
    d1 = 1 if d1 == 'R' else (-1 if d1 == 'L' else 'N')
    d2 = 1 if d2 == 'D' else (-1 if d2 == 'U' else 'N')
    if d1 == 'N' or d2 == 'N':
        print("ERROR in assigning point curves, invalid directions")
    d3 = -d1
    d4 = -d2
    
    for i in range(0, point_num-1):
        y = point[i][0]
        x = point[i][1]
        if i == 0:
            print(f"Starting at y{y}/x{x}")
        dy = 0 # difference in position from previous to current
        dx = 0
        if i != 0: # check if point was found at prev iteration
            prevPoint = point[i-1]
            dy = y - prevPoint[0]
            dx = x - prevPoint[1]
        
        # check the neighboring pixels in a + shape for a non-0 value
        y1 = y+d2 # position of pixel to be checked
        x1 = x+d1
        neighbor1 = img[y,x1]
        neighbor2 = img[y1,x]
        y2 = y+d4
        x2 = x+d3
        neighbor3 = img[y,x2]
        neighbor4 = img[y2,x]
        
        # check if the neighbor is non-0 and is not the previous point,
        # and make that the grid coordinates of the next point, ensuring that
        # only one set of coordinates is chosen.
        found = False
        if neighbor1 == 1 and d1 != -dx and found == False:
            next_grid = [y,x1]
            found = True
        if neighbor2 == 1 and d2 != -dy and found == False:
            next_grid = [y1,x]
            found = True
        if neighbor3 == 1 and d3 != -dx and found == False:
            next_grid = [y,x2]
            found = True
        if neighbor4 == 1 and d4 != -dy and found == False:
            next_grid = [y2,x]
            found = True
        
        point[i+1] = next_grid
    
    return point;

def normalize(img, top_val):
    mn = np.min(img)
    mx = np.max(img)
    norm_img = 255 * (img - mn) / (mx - mn)
    #rows = len(img)
    #cols = img[0].size
    #norm_img = np.zeros((rows,cols))
    #for row in range(0, rows):
     #   for col in range(0, cols):
      #      norm_img[row,col] = (img[row,col] - mn) * (1 / (mx - mn))
    
    return norm_img;

"""def compute_curvature(pointcurve_data):
    P = pointcurve_data # array of points each with a y-x coordinate.
    # e.g. P[0] = [4][12] point 0 located at y=4 and x=12
    Pcount = len(P)

    K = [] # value of curvature at each point indexed respectively.
    dx = [] # x-direction derivatives (Δx/(P[+1] - P[-1])) discrete center-method.
    dy = [] # y-direction (Δy/(P[+1] - P[-1])).
    d2x = [] # x-direct. double derivatives ((dx[+1] - dx[-1])/(P[+1] - P[-1]))
    d2y = [] # y-direct. double derivatives (Δdx/ΔP).

    # example diagram of derivatives for this function
    #
    #       |<------- d2x ------->|
    #       |                     |
    #  |<--- dx_L ---->|<---- dx_R --->|
    #  |               |               |
    #  |     |<----- dx_C ----->|      |
    #  |     |                  |      |
    #  |x=12 | x=12  x=11  x=12 |  x=11|
    #  P[-2]  P[-1]  P[0]  P[+1]  P[+2]

    # find first derivatives
    for i in range(0, Pcount): # iterate through points 0 to N
        if i < Pcount - 1: # for all points 0 to 2nd-to-last point
            val_diff = P[i+1] - P[i-1]
            dy.append(val_diff[0]/2)
            dx.append(val_diff[1]/2)
        else: # for the last point, to avoid index overflow
            val_diff = P[0] - P[i-1]
            dy.append(val_diff[0]/2)
            dx.append(val_diff[1]/2)
            #print(f"last point: x' = {dx[i]} y' = {dy[0]}")
    dx = np.array(dx)
    dy = np.array(dy)
    
    # find second derivatives
    for i in range(0, Pcount): # iterate through points 0 to N
        if i < Pcount - 1: # for all points 0 to 2nd-to-last point
            delta_dx = dx[i+1] - dx[i-1]
            delta_dy = dy[i+1] - dy[i-1]
            d2x.append(delta_dx / 2)
            d2y.append(delta_dy / 2)
        else: # for the last point, to avoid index overflow
            #val_diff = P[0] - P[i-1]
            delta_dx = dx[0] - dx[i-1]
            delta_dy = dy[0] - dy[i-1]
            d2x.append(delta_dx / 2)
            d2y.append(delta_dy / 2)
            #print(f"x': P[+1]={dx[0]} P[-1]={dx[i-1]}")
            #print(f"y': P[+1]={dy[0]} P[-1]={dy[i-1]}")
            #print(f"Δdy: {delta_dy}, Δdx: {delta_dx}")
            #print(f"P[last]: x'' = {d2x[i]} y'' = {d2y[i]}")
    d2x = np.array(d2x)
    d2y = np.array(d2y)
    
    # compute curvature in accordance with formula for K
    #           x' y" - y' x"
    # K = --------------------------
    #      [ (x')^2 + (y')^2 ]^(3/2)
    for i in range(0, Pcount):
        top = (dx[i] * d2y[i]) - (dy[i] * d2x[i])
        bot = ((dx[i]**2) + (dy[i]**2))**(3/2)
        K.append(top/bot)
    K = np.array(K, dtype=np.float16)
    
    return K;"""

def create_curvature_img(curvatures, pointcurve_map, base_img):
    K = np.array(curvatures) # list of curvatures corresponding to the points
    P = np.array(pointcurve_map) # list of points in-order along a curve
    base = np.array(base_img)
    # format of P: P[0] = [row][col]
    # format of output: newImg[0] = [row][col][K]
    # in other words: pixel at P[0] is located at [row][col] of image and...
    # ... has a value [K] that represents the curvature at that point.
    
    # create mask of curvatures to apply to base image
    M = [] # output image mask
    Pcount = len(P)
    for i in range(0, Pcount):
        M.append([P[i][0], P[i][1], K[i]])
    M = np.array(M, dtype=np.float16)
    #print("First point: ", M[0])
    #print("Last point: ", M[-1])
    
    # apply mask to base image
    rows = len(base)
    cols = base[0].size
    newImg = np.zeros((rows,cols), dtype=np.float16)
    #print(f"rows/cols = {rows}/{cols}")
    new_pixel_count = 0
    print(f"row = {M[0][0].astype(int)} col = {M[0][1].astype(int)} k = {M[0][2]}")
    for row in range(0, rows):
        for col in range(0, cols):
            for i in range(0, Pcount):
                if row == M[i][0].astype(int) and col == M[i][1].astype(int):
                    newImg[row][col] = M[i][2]
                    new_pixel_count = new_pixel_count + 1
    print("New pixels written: ", new_pixel_count)
    
    return newImg;
    
"""
# Testbench code
testimg = iio.imread('test_bean.png')
testimg = ski.color.rgb2gray(testimg)
bigimg = expand_canvas(testimg, 5)
edgeimg = ski.filters.roberts(bigimg)
edgeimg[edgeimg != 0] = 1
edgeimg = np.array(edgeimg, dtype=int)
pointcurve = assign_point_curves(edgeimg)
curvature_k = compute_curvature(pointcurve)
curveimg = create_curvature_img(curvature_k, pointcurve, edgeimg)
curve_and_bean = curveimg + edgeimg
plt.imshow(curve_and_bean, cmap='nipy_spectral')
"""