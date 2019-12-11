# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:26:31 2019

@author: vince
"""

import derivatives
from pylab import *
from numpy import *
from scipy.ndimage import filters


def compute_harris_response(im,sigma=3):
    
    imx, imy, mag = derivatives.gradient_gaussian(im, sigma)
    
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)
    
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    
    return Wdet / Wtr


def get_harris_points(harrisim, min_dist = 10, threshold=0.1):
    """ Return corners from a Harris response image, min_dist is the minimum 
    number of pixels separating corners and image boundary."""
    
    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    
    # get coordinates of candidates
    coords = array(harrisim_t.nonzero()).T
    
    # ...and their values
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    
    # sort candidates
    index = argsort(candidate_values)
    
    # store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0], coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist), (coords[i,1]-min_dist):(coords[i,1]+min_dist),
                              (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    
    return filtered_coords


    def plot_harris_points(image, filtered_coords):
        """ Plots corners found in image. """
        
        figure()
        gray()
        imshow(image)
        plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords], '*')
        axis('off')
        show()