import cv2
import os
import threeD
import matches_selection
import camera
import sift


dirname = os.path.dirname(__file__)
data_path = os.path.join(dirname, 'dataset')

filename = os.path.join(data_path, 'building/IMG_4288.jpg')
im1 = cv2.imread(filename, 0)

filename = os.path.join(data_path, 'building/IMG_4290.jpg')
im2 = cv2.imread(filename, 0)

img_list = [im1, im2]

# initialisation of the SIFT descriptor
sift_dtc = sift.sift(nfeatures=5000, sigma=5)

kp_list, matches_list = sift_dtc.fit_multiple(img_list, lowe_factor=.75)

# RANSAC with DLT algorithm to select robust matches
matches_s, H = matches_selection.ransac_matches_selection(kp_list[0], kp_list[1], matches_list[0],
                                                          match_threshold=0.2*im1.shape[0])

# plot the matches and the two images
matches_selection.plot_matches(im1, im2, kp_list[0], kp_list[1], matches_s)

# focal length deduced from the metadata of the images and the size of the captor of the camera
fx = im1.shape[0]*24/35.8
fy = im1.shape[1]*24/23.9

# generates an instinsic matrix with fx, fy and 1 on the diagonal and col/2, row/2 on the last column
K = camera.calibration(im1.shape, fx, fy)

# there is also a reconstruction_bis which uses a different approach and a reconstruction_uncalibrated
X = threeD.reconstruction(img_list[0], img_list[1], kp_list[0], kp_list[1], matches_s, K)
# the red points are the keypoints from SIFT, the blue points are the 3D points reprojected in the image plan using the camera matrix