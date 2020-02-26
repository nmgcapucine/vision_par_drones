import numpy as np
import cv2


class sift(object):

    def __init__(self, nfeatures=5000, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
        self.detector = cv2.xfeatures2d_SIFT.create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers,
                                                    contrastThreshold=contrastThreshold,
                                                    edgeThreshold=edgeThreshold, sigma=sigma)

    def fit_multiple(self, img_list, lowe_factor=.7):
        kp_list = []
        descriptor_list = []
        matches_list = []

        for img in img_list:
            kp = self.detector.detect(img)
            kp_list.append(kp)
            kp, des = self.detector.compute(img, kp)
            descriptor_list.append(des)

        bf = cv2.BFMatcher_create(cv2.NORM_L2)

        n = len(img_list)
        for i in range(n-1):
            matches = bf.knnMatch(descriptor_list[i], descriptor_list[i+1], k=2)
            good = []
            for m, n in matches:
                if m.distance < lowe_factor * n.distance:
                    good.append([m])
            matches_list.append(np.array(good)[:, 0])

        return kp_list, matches_list