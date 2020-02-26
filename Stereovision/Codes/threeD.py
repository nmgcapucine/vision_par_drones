import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import homography
import sfm
import camera


def plot_3D_points(points3D, P1, P2):
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = Axes3D(fig)
    ax.plot(points3D[0], points3D[1], points3D[2], 'k.')
    ax.plot([P1[0, 3], P2[0, 3]], [P1[1, 3], P2[1, 3]], [P1[2, 3], P2[0, 3]], 'r.')
    plt.show()


def reconstruction(im1, im2, kp1, kp2, matches, K):

    pts1 = homography.make_homog(np.transpose(np.array([kp1[m.queryIdx].pt for m in matches])))
    pts2 = homography.make_homog(np.transpose(np.array([kp2[m.trainIdx].pt for m in matches])))

    x1n = np.dot(np.linalg.inv(K), pts1)
    x2n = np.dot(np.linalg.inv(K), pts2)

    # estimate E with RANSAC (since we normalized with K^-1, we will get E with the normalized 8 pts algo.)
    model = sfm.RansacModel()
    E, inliers = sfm.F_from_ransac(x1n, x2n, model,  maxiter=5000, match_threshold=1e-5)
    E = sfm.compute_fundamental_normalized(x1n[:, inliers], x2n[:, inliers])

    # compute camera matrices (P2 will be list of four solutions)
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    P2 = sfm.compute_P_from_essential(E)

    # pick the solution with points in front of cameras
    ind = 0
    maxres = 0

    for i in range(4):
        # triangulate inliers and compute depth for each camera
        X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[i])
        # X = sfm.triangulate(x1n, x2n, P1, P2[i])
        d1 = np.dot(P1, X)[2]
        d2 = np.dot(P2[i], X)[2]
        if np.sum(d1 > 0)+np.sum(d2 > 0) > maxres:
            maxres = np.sum(d1 > 0)+sum(d2 > 0)
            ind = i
            infront = (d1 > 0) & (d2 > 0)


    # triangulate inliers and remove points not in front of both cameras
    X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[ind])
    X = X[:, infront]

    # 3D plot
    plot_3D_points(X, P1, P2[ind])

    cam1 = camera.Camera(P1)

    cam2 = camera.Camera(P2[ind])
    print(P2[ind])
    x1p = cam1.project(X)
    x2p = cam2.project(X)

    # reverse K normalization
    x1p = np.dot(K, x1p)
    x2p = np.dot(K, x2p)

    x1 = pts1[:, inliers]
    x2 = pts2[:, inliers]
    x1 = x1[:, infront]
    x2 = x2[:, infront]

    plt.figure()
    plt.imshow(im1)
    plt.gray()
    plt.plot(x1p[0], x1p[1], 'o')
    plt.plot(x1[0], x1[1],'r.')
    plt.axis('off')

    plt.figure()
    plt.imshow(im2)
    plt.gray()
    plt.plot(x2p[0], x2p[1], 'o')
    plt.plot(x2[0], x2[1], 'r.')
    plt.axis('off')
    plt.show()

    error1 = np.mean(np.linalg.norm(x1 - x1p, axis=0))
    error2 = np.mean(np.linalg.norm(x2 - x2p, axis=0))

    print('error 1 : ', error1)
    print('error 2 : ', error2)

    return X


def reconstruction_bis(im1, im2, kp1, kp2, matches, K):

    pts1 = homography.make_homog(np.transpose(np.array([kp1[m.queryIdx].pt for m in matches])))
    pts2 = homography.make_homog(np.transpose(np.array([kp2[m.trainIdx].pt for m in matches])))


    x1n = np.dot(np.linalg.inv(K), pts1)
    x2n = np.dot(np.linalg.inv(K), pts2)

    T1 = sfm.H_norm(x1n)
    T2 = sfm.H_norm(x2n)

    x1n = np.dot(T1, x1n)
    x2n = np.dot(T2, x2n)

    # estimate E with RANSAC
    model = sfm.RansacModel()
    E, inliers = sfm.F_from_ransac(x1n, x2n, model,  maxiter=5000, match_threshold=1e-6)
    E = sfm.compute_fundamental(x1n[:, inliers], x2n[:, inliers])

    # de-normalize
    x1n = np.dot(np.linalg.inv(T1), x1n)
    x2n = np.dot(np.linalg.inv(T2), x2n)

    E = np.dot(T1.T, np.dot(E, T2))
    E = E/E[2,2]

    # compute camera matrices (P2 will be list of four solutions)
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    P2 = sfm.compute_P_from_essential(E)

    # pick the solution with points in front of cameras
    ind = 0
    maxres = 0


    for i in range(4):
        # triangulate inliers and compute depth for each camera
        X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[i])
        # X = sfm.triangulate(x1n, x2n, P1, P2[i])
        d1 = np.dot(P1, X)[2]
        d2 = np.dot(P2[i], X)[2]
        if np.sum(d1 > 0)+np.sum(d2 > 0) > maxres:
            maxres = np.sum(d1 > 0)+sum(d2 > 0)
            ind = i
            infront = (d1 > 0) & (d2 > 0)


    # triangulate inliers and remove points not in front of both cameras
    X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[ind])
    X = X[:, infront]

    # 3D plot
    plot_3D_points(X, P1, P2[ind])

    cam1 = camera.Camera(P1)

    cam2 = camera.Camera(P2[ind])
    print(P2[ind])
    x1p = cam1.project(X)
    x2p = cam2.project(X)

    # reverse K normalization
    x1p = np.dot(K, x1p)
    x2p = np.dot(K, x2p)

    x1 = pts1[:, inliers]
    x2 = pts2[:, inliers]
    x1 = x1[:, infront]
    x2 = x2[:, infront]

    plt.figure()
    plt.imshow(im1)
    plt.gray()
    plt.plot(x1p[0], x1p[1], 'o')
    plt.plot(x1[0], x1[1], 'r.')
    plt.axis('off')

    plt.figure()
    plt.imshow(im2)
    plt.gray()
    plt.plot(x2p[0], x2p[1], 'o')
    plt.plot(x2[0], x2[1], 'r.')
    plt.axis('off')
    plt.show()

    print(np.hypot(x1 - x1p).shape)
    error1 = np.mean(np.hypot(x1 - x1p))
    error2 = np.mean(np.hypot(x2 - x2p))

    print('error 1 : ', error1)
    print('error 2 : ', error2)

    return X


def reconstruction_uncalibrated(im1, im2, kp1, kp2, matches):

    pts1 = homography.make_homog(np.transpose(np.array([kp1[m.queryIdx].pt for m in matches])))
    pts2 = homography.make_homog(np.transpose(np.array([kp2[m.trainIdx].pt for m in matches])))

    T1 = sfm.H_norm(pts1)
    T2 = sfm.H_norm(pts2)

    x1n = np.dot(T1, pts1)
    x2n = np.dot(T2, pts2)
    #x1n = pts1
    #x2n = pts2

    # estimate F with 8 pt algorithm
    model = sfm.RansacModel()
    F, inliers = sfm.F_from_ransac(x1n, x2n, model, maxiter=5000, match_threshold=1e-5)
    F = sfm.compute_fundamental(x1n[:, inliers], x2n[:, inliers])

    #denormalization
    x1n = np.dot(np.linalg.inv(T1), x1n)
    x2n = np.dot(np.linalg.inv(T2), x2n)

    F = np.dot(T1.T, np.dot(F, T2))
    F = F/F[2,2]

    print(F)
    print(inliers.shape)
    # compute camera matrices (P2 will be list of four solutions)
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    P2 = sfm.compute_P_from_fundamental(F)
    print(P2)
    # triangulate inliers and remove points not in front of both cameras
    X = sfm.triangulate(x1n, x2n, P1, P2)
    d1 = np.dot(P1, X)[2]
    d2 = np.dot(P2, X)[2]
    infront = (d1 > 0) & (d2 > 0)
    print(infront)
    #X = X[:, infront]

    # 3D plot
    plot_3D_points(X, P1, P2)

    cam1 = camera.Camera(P1)
    cam2 = camera.Camera(P2)

    x1p = cam1.project(X)
    x2p = cam2.project(X)

    #x1n = np.dot(np.linalg.inv(T1), x1n)
    #x2n = np.dot(np.linalg.inv(T2), x2n)
    #x1p = np.dot(np.linalg.inv(T1), x1p)
    #x2p = np.dot(np.linalg.inv(T2), x2p)

    x1 = pts1
    x2 = pts2
    #x1 = x1[:, infront]
    #x2 = x2[:, infront]

    plt.figure()
    plt.imshow(im1)
    plt.gray()
    plt.plot(x1p[0], x1p[1], 'o')
    plt.plot(x1[0], x1[1], 'r.')
    plt.axis('off')

    plt.figure()
    plt.imshow(im2)
    plt.gray()
    plt.plot(x2p[0], x2p[1], 'o')
    plt.plot(x2[0], x2[1], 'r.')
    plt.axis('off')
    plt.show()

    return X