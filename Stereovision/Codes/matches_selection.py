import numpy as np
import homography
from matplotlib import pyplot as plt


def ransac_matches_selection(kp1, kp2, matches, match_threshold=100):

    model = homography.RansacModel()

    pts1 = homography.make_homog(np.transpose(np.array([kp1[m.queryIdx].pt for m in matches])))
    pts2 = homography.make_homog(np.transpose(np.array([kp2[m.trainIdx].pt for m in matches])))

    H, inliers = homography.H_from_ransac(pts1, pts2, model, maxiter=1000, match_threshold=match_threshold)

    return matches[inliers], H


def plot_matches(img1, img2, kp1, kp2, matches):
    pts1 = homography.make_homog(np.transpose(np.array([kp1[m.queryIdx].pt for m in matches])))
    pts2 = homography.make_homog(np.transpose(np.array([kp2[m.trainIdx].pt for m in matches])))

    img3 = appendimages(img1, img2)
    #img3 = np.vstack((img3, img3))
    plt.imshow(img3)
    plt.gray()

    cols1 = img1.shape[1]

    for i in range(len(matches)):
        plt.plot([pts1[0][i], pts2[0][i]+cols1], [pts1[1][i], pts2[1][i]], 'c', linewidth=0.5, linestyle='--')
        plt.axis('off')

    plt.plot(pts1[0], pts1[1], 'r.')
    plt.axis('off')

    plt.plot(pts2[0]+cols1, pts2[1], 'r.')
    plt.axis('off')
    plt.show()


def appendimages(im1, im2):
    """ Return a new image that appends the two images side-by-side. """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.

    return np.concatenate((im1, im2), axis=1)