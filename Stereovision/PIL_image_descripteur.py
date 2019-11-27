# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:57:05 2019

@author: vince
"""

import imtools
import harris
from PIL import Image
from pylab import *
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
from scipy.misc import ascent

resolution = (330,250)


import sift

imname = 'IMG_3352.jpg'
im1 = imtools.Load_grey_image(imname, resolution)

sift.process_image(imname,'IMG_3352.sift')
l1,d1 = sift.read_features_from_file('IMG_3352.sift')

figure()
gray()
sift.plot_features(im1,l1,circle=True)
show()

#im1 = imtools.Load_grey_image('IMG_3352.jpg', resolution)
#im2 = imtools.Load_grey_image('IMG_3353.jpg', resolution)
#
#wid = 5
#harrisim = harris.compute_harris_response(im1,5)
#filtered_coords1 = harris.get_harris_points(harrisim,wid+1)
#d1 = harris.get_descriptors(im1,filtered_coords1,wid)
#
#harrisim = harris.compute_harris_response(im2,5)
#filtered_coords2 = harris.get_harris_points(harrisim,wid+1)
#d2 = harris.get_descriptors(im2,filtered_coords2,wid)
#
#print('starting matching')
#matches = harris.match_twosided(d1,d2)
#
#figure()
#gray()
#harris.plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches)
#show()


#im = ascent()
#im2, cdf = imtools.histeq(im)

#im2_list = array([zeros(im.shape)]*5)

#for i in range(5):
#    im2_list[i] = filters.gaussian_filter(im, (i+1)*5)



#filtered_coords = test_orientation.compute_response_oriented(im, 3, 0, 50, 10)
#figure()
#gray()
#imshow(im)
#plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords], '*')
#axis('off')
#savefig('oriented_points.jpg')
#show()

#harrisim = harris.compute_harris_response(im)
#filtered_coords = harris.get_harris_points(harrisim, 20)
##harris.plot_harris_points(im, filtered_coords)
#figure()
#gray()
#imshow(im)
#plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords], '*')
#axis('off')
#savefig('harris.jpg')
#show()
#print(len(filtered_coords))




#x = [100, 100, 400, 400]
#y = [200, 500, 200, 500]

#plot(x, y, 'r*')
#plot(x[:2], y[:2], 'go-')
#

#show()


#figure()
#hist(im.flatten(), 128)
#show()
#
#figure()
#hist(im2.flatten(), 128)
#show()
#print(im.shape)
#print(im.dtype)

