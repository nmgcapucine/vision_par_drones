# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:43:01 2019

@author: vince
"""

import derivatives
from PIL import Image
from pylab import *
import numpy as np
from scipy.ndimage import filters
from scipy.misc import imsave

def compute_response_oriented(im, sigma=3, angle = 0, threshold = 10, min_dist = 10):
    
    im_main, im_orth, mag = derivatives.gradient_direction_gaussien(im, sigma, angle)
    
    diff = zeros(im.shape)
    
    Xmax, Ymax = im.shape
    for x in range(Xmax):
        for y in range(Ymax):    
            if im_orth[x,y] == 0:
                diff[x,y] = threshold + 1
            else:
                diff[x,y] = abs(im_main[x,y])/abs(im_orth[x,y])
    
    corner_threshold = im_main.max() * 0.1
    
    select_ratio = (diff > threshold) * 1
    
    select_thresh = (im_main > corner_threshold) * 1
    
    diff_over_thresh = select_ratio * select_thresh
    # get coordinates of candidates
    coords = np.array(diff_over_thresh.nonzero()).T
    
    # ...and their values
    candidate_values = [diff[c[0],c[1]] for c in coords]
    
    # sort candidates
    index = np.argsort(candidate_values)
    
    # store allowed point locations in array
    allowed_locations = np.zeros(diff.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0], coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist), (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    
    return filtered_coords