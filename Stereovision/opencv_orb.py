import numpy as np
import matches_selection
import cv2
from matplotlib import pyplot as plt
import os

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(dirname, 'Datasets')

# Image name between 3434 and 3444 for tea_scene
filename = os.path.join(data_path, 'tea_scene/IMG_3434.JPG')
im1 = cv2.imread(filename, 0)
filename = os.path.join(data_path, 'tea_scene/IMG_3437.JPG')
im2 = cv2.imread(filename, 0)


def orb_kp(img):
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.3, nlevels=10, edgeThreshold=31, firstLevel=0, WTA_K=2,
                         patchSize=31, fastThreshold=20)

    # find the key points with ORB
    kp = orb.detect(img)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des


def match_points(des1, des2):
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def plot_matches(img1, img2, kp1, kp2, matches):
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

    plt.figure()
    plt.imshow(img3)
    plt.show()


class orb(object):

    def __init__(self, nfeatures=1000, scaleFactor=1.3, nlevels=10, edgeThreshold=31, firstLevel=0, WTA_K=2,
                patchSize=31, fastThreshold=20):
        self.detector = cv2.ORB_create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K,
                                       patchSize, fastThreshold)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    def fit(self, img1, img2):
        kp1, kp2 = self.detector.detect(img1), self.detector.detect(img2)
        kp1, des1 = self.detector.compute(img1, kp1)
        kp2, des2 = self.detector.compute(img2, kp2)

        matches = self.matcher.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        return np.array(kp1), np.array(kp2), np.array(des1), np.array(des2), np.array(matches)

    def plot_matches(self, img1, img2, n_matches = None):
        kp1, kp2 = self.detector.detect(img1), self.detector.detect(img2)
        kp1, des1 = self.detector.compute(img1, kp1)
        kp2, des2 = self.detector.compute(img2, kp2)
        matches = self.matcher.match(des1, des2)

        if n_matches is None:
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        else:
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:n_matches], None, flags=2)

        plt.figure()
        plt.imshow(img3)
        plt.show()
