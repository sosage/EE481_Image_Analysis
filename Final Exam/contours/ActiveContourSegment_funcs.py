"""
Image Analysis - Final Exam - Question 3 - Active Countour
Segmentation Custom Functions File
Created on Thu Dec  11 19:19:00 2025

@author: S.W. Marsden
"""

import numpy as np
from skimage import color, segmentation
from skimage.restoration import denoise_tv_chambolle
#from scipy.ndimage import convolve
import matplotlib.pyplot as plt

# setup the plot coordinates for plotting multiple images
def initialize_plot_locations(grid):
    # Create plot location coordinates. For a 3x3 grid of plots,
    # there are 9 locations each with a (row, col) coordinate.
    # PLOC[0] = (1,1); PLOC[1] = (1,2); PLOC[2] = (1,3)
    # PLOC[3] = (2,1); PLOC[4] = (2,2); PLOC[5] = (2,3)
    # PLOC[6] = (3,1); PLOC[7] = (3,2); PLOC[8] = (3,3)
    rows = grid[0]
    cols = grid[1]
    PLOC = np.empty( ((rows * cols), 2) ).astype(np.uint8)
    idx = 0
    for row in range(0, rows):
        for col in range(0, cols):
            PLOC[idx] = (row+1, col+1)
            idx += 1
    idx = 0
    
    return PLOC;

# setup for plotting multiple images
def initialize_multiplot(plot_grid, text_size=8, res=800):
    plt.rc('font', size=text_size)
    fig, ax = plt.subplots(plot_grid[0], plot_grid[1],
                           dpi=res, constrained_layout=True)
    return ax;

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
    
    if plotType == 'contour':
        ax[rowI][colI].plot(data[:, 1], data[:, 0], '-b', lw=3)
    
    if gridshare == 1:
        ax[rowI][colI].sharex(ax[0][0])
        ax[rowI][colI].sharey(ax[0][0])
   
    return 0;

# grayscale 2, 3, or 4 channel images
def convert_to_gray(img):
    img_dimension = len(img.shape)
    if img_dimension == 2:
        return img;
    elif img_dimension == 3:
        img = color.rgb2gray(img[:,:,0:3])
        """minimum_val = np.min(img)
        maximum_val = np.max(img)
        img = (img - minimum_val) / (maximum_val - minimum_val)
        img = np.array((255 * img))
        """
        # if there are more than 3 color params (RGBA or CMYK),
        # only 0 thru 3 are used.
        return img;
    else:
        print("ERROR while grayscaling.")
        return 1;

# take a square section from top left of an image for detail analysis
def snip(img, sizeX=300, sizeY=300):
    img_snip = img[:sizeY,:sizeX]
    return img_snip;

""" from skimage.restoration import denoise_tv_chambolle """
# denoise the image using the total var. chambolle method
def total_var_smooth(roughImg, intensity=0.5, loops=100):
    smoothImg = denoise_tv_chambolle(roughImg,
                                     weight=intensity,
                                     eps=0.0002,
                                     max_num_iter=loops)
    return smoothImg;

""" from skimage import segmentation """
# segment the image as IN or OUT using CV-ACWE method.
def chan_vese_segment(img, iters=10):
    segments = segmentation.chan_vese(img,
                                      max_num_iter=iters)
    
    return segments;

"""
# take an image and compute the active contour curve
def find_contour(img):
    s = np.linspace(0, 2 * np.pi, 400) # create the initial
    r = 200 + 150 * np.sin(s)          # contour circle.
    c = 220 + 150 * np.cos(s)
    init = np.array([r, c]).T
    
    contour = segmentation.active_contour(img, init,
                                          alpha=0.015,
                                          beta=10,
                                          gamma=0.001,
                                          max_num_iter=100)
    
    return contour;
"""

"""
def compute_LuKan(patch, It):
    "" Ix * u + Iy * v + It = 0   
        u and v are unknowns, I is from the images.
        Ix = dI/dx   Iy = dI/dy   It = dI/dt = I1 - I2
        u = dx/dt   v = dy/dt
    ""
    # create the kernels to compute gaussian derivatives
    rows = patch.shape[0]
    cols = patch.shape[1]
    gauss_xkernel = np.zeros(patch.shape)
    for i in range(gauss_xkernel.shape[-1]):
        gauss_xkernel[:, i] = -((cols-1)/2) + i
    gauss_ykernel = np.zeros(patch.shape)
    for i in range(gauss_ykernel.shape[-1]):
        gauss_ykernel[i, :] = -((rows-1)/2) + i
    # compute gaussian derivatives of the image-patch to
    # yield Ix and Iy.
    mn = np.min(patch)
    mx = np.max(patch)
    if mx - mn != 0:
        patch_norm = (patch - mn) / (mx - mn)
    else:
        patch_norm = (patch - mn) / (mx - mn + 0.00001)
    Ix = convolve(patch_norm, gauss_xkernel)
    Iy = convolve(patch_norm, gauss_ykernel)
    # turn Ix, Iy, and It into vectors for easier manipulation.
    Ix_v = np.zeros((rows*cols, 1))
    Iy_v = np.zeros((rows*cols, 1))
    It_v = np.zeros((rows*cols, 1))
    idx = 0
    for row in range(0,rows):
        for col in range(0, cols):
            Ix_v[idx] = Ix[row][col]
            Iy_v[idx] = Iy[row][col]
            It_v[idx] = It[row][col]
            idx += 1
    # create the A and b matrix for solving
    A = np.zeros((patch.size, 2))
    b = np.zeros((patch.size, 1))
    for row in range(0, A.shape[0]):
        A[row][0] = Ix_v[row]
        A[row][1] = Iy_v[row]
    for row in range(0, b.shape[0]):
        b[row][0] = It_v[row]
    "" [ Ix(p1) Iy(p1) ]             [ It(p1) ]
	    [ Ix(p2) Iy(p2) ] [ u ]       [ It(p2) ]
	    [ ...    ...    ] [ v ] = (-) [ ...    ]
	    [ Ix(pN) Iy(pN) ]             [ It(pN) ]
    ""
    # solve for u and v (x̂ = [u v]) to find motion vector.
    # x̂ = arg(x)min || Ax - b ||^2   ~~~>>   A^T A x̂ = A^T b
    # using least squares solution to solve.
    A_prime = np.zeros((2,2))
    b_prime = np.zeros((2,1))
    for i in range(0, A.shape[0]):
        A_prime[0,0] = A_prime[0,0] + (Ix_v[i] * Ix_v[i]) # Σ{ Ix Ix }, p ∈ P
        A_prime[0,1] = A_prime[0,1] + (Ix_v[i] * Iy_v[i]) # Σ{ Ix Iy }
        A_prime[1,0] = A_prime[1,0] + (Iy_v[i] * Ix_v[i]) # Σ{ Iy Ix }
        A_prime[1,1] = A_prime[1,1] + (Iy_v[i] * Iy_v[i]) # Σ{ Iy Iy }
        b_prime[0,0] = b_prime[0,0] + (Ix_v[i] * It_v[i]) # Σ{ Ix It }
        b_prime[1,0] = b_prime[1,0] + (Iy_v[i] * It_v[i]) # Σ{ Iy It }
    ""            A^T A              x̂  =         A^T b
        [                         ]             [            ]
        [ Σ{ Ix Ix }   Σ{ Ix Iy } ]             [ Σ{ Ix It } ]
	    [ p ∈ P        p ∈ P    ]             [ p ∈ P     ]
        [                         ] [ u ]       [            ]
        [ Σ{ IyIx }    Σ{ Iy Iy } ] [ v ] = (-) [ Σ{ Iy It } ]
	    [ p ∈ P        p ∈ P    ]             [ p ∈ P     ]
        [                         ]             [            ]
        where p ∈ P is all pixels p in the patch P.
    ""
    # calculate motion vector x̂ = [u v]
    # x̂ = (A^T A)^(-1) A^T b
    if np.linalg.det(A_prime) != 0:
        x_v = np.linalg.inv(A_prime) @ b_prime
    else:
        x_v = np.linalg.pinv(A_prime) @ b_prime
    u = x_v[0,0]
    v = x_v[1,0]
    
    return u, v;"""
    
    
    
    