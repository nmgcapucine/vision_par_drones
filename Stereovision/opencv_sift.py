import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import matches_selection

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(dirname, 'Datasets')

# Image name between 3434 and 3444 for tea_scene
filename = os.path.join(data_path, 'selected/Image0004.JPG')
im1 = cv2.imread(filename, 0)
filename = os.path.join(data_path, 'selected/Image0011.JPG')
im2 = cv2.imread(filename, 0)

filename = os.path.join(data_path, 'selected/Image0004_crop.JPG')
im3 = cv2.imread(filename, 0)
filename = os.path.join(data_path, 'selected/Image0011_crop.JPG')
im4 = cv2.imread(filename, 0)



def sift_kp(img):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d_SIFT.create(nfeatures=1000, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10,
                                       sigma=1.6)

    kp = sift.detect(img)

    kp, des = sift.compute(img, kp)
    return kp, des


def match_points_sift_knn(des1, des2):
    bf = cv2.BFMatcher_create(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])

    return good


def plot_matches_sift(img1, img2):
    kp1, des1 = sift_kp(img1)
    kp2, des2 = sift_kp(img2)

    bfm = cv2.BFMatcher_create(cv2.NORM_L2)

    matches = match_points_sift_knn(des1, des2)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=4)

    plt.figure()
    plt.imshow(img3)
    plt.show()


class sift(object):

    def __init__(self, nfeatures=5000, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
        self.detector = cv2.xfeatures2d_SIFT.create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers,
                                                    contrastThreshold=contrastThreshold,
                                                    edgeThreshold=edgeThreshold, sigma=sigma)

    def fit(self, img1, img2, lowe_factor =.7):
        kp1 = self.detector.detect(img1)
        kp2 = self.detector.detect(img2)

        kp1, des1 = self.detector.compute(img1, kp1)
        kp2, des2 = self.detector.compute(img2, kp2)

        bf = cv2.BFMatcher_create(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < lowe_factor * n.distance:
                good.append([m])

        return np.array(kp1), np.array(kp2), np.array(des1), np.array(des2), np.array(good)[:, 0]
