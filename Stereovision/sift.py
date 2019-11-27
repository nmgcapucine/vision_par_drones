# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:22:15 2019

@author: vince
"""

from PIL import Image
import os
from numpy import *

def process_image(imagename,resultname,params="--edge-tresh 10 --peak-tresh 5"):
    """ Process an image and save the result in a file. """
    
    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'
        
    cmmd = str("sift "+imagename+" --output="+resultname+" "+params)
    os.system(cmmd)
    print('processed', imagename, 'to', resultname)

def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """
    
    f = loadtxt(filename)
    return f[:,:4],f[:,4:] # feature locations, descriptors

def write_features_to_file(filename,locs,desc):
    """ Save feature location and descriptor to file. """
    savetxt(filename,hstack((locs,desc)))

def plot_features(im,locs,circle=False):
    """ Show image with features. input: im (image as array),
    locs (row, col, scale, orientation of each feature). """
    
    def draw_circle(c,r):
        t = arange(0,1.01,.01)*2*pi
        x = r*cos(t) + c[0]
        y = r*sin(t) + c[1]
        plot(x,y,'b',linewidth=2)
        
    imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2],p[2])
    else:
        plot(locs[:,0],locs[:,1])
    axis('off')