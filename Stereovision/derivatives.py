# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:36:09 2019

@author: vince
"""

from PIL import Image
from pylab import *
from numpy import *
from scipy.ndimage import filters
import math

def gradient_sobel(img):
    imx = zeros(img.shape)
    filters.sobel(img,1,imx)
    
    imy = zeros(img.shape)
    filters.sobel(img, 0, imy)
    
    magnitude = sqrt(imx**2+imy**2)
    
    return imx, imy, magnitude

def gradient_gaussian(img, sigma=3):
    imx = zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (0,1), imx)
    
    imy = zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (1, 0), imy)
    
    magnitude = sqrt(imx**2+imy**2)
    
    return imx, imy, magnitude

def gradient_direction_gaussien(img, sigma = 3, angle = 0):
    """ direction est un angle en radiansformé entre la direction verticale et
    la direction souhaitée"""
    
    main_dir = [math.cos(angle), math.sin(angle)]
    
    imx = zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (0,1), imx)
    
    imy = zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (1, 0), imy)
    
    orth_dir = [main_dir[1], -main_dir[0]]
    
    im_main = imx * main_dir[0] + imy * main_dir[1]
    im_orth = imx * orth_dir[0] + imy * orth_dir[1]
    
    magnitude = sqrt(im_main**2+im_orth**2)
    
    return im_main, im_orth, magnitude