import numpy as np
import opencv_orb
import homography


def homog_matches_selection(kp1, kp2, matches, epsilon=100, n_test=100):

    n_matches = len(matches)

    pts1 = homography.make_homog(np.transpose(np.array([kp1[m.queryIdx].pt for m in matches])))
    pts2 = homography.make_homog(np.transpose(np.array([kp2[m.trainIdx].pt for m in matches])))

    kept_index_list = []

    for n in range(n_test):
        selected_matches_idx = np.random.randint(n_matches, size=4)

        fp = pts1[:, selected_matches_idx]
        tp = pts2[:, selected_matches_idx]

        H = homography.H_from_points(fp, tp)

        Hpts1 = homography.normalize(H.dot(pts1))

        dist = np.sqrt(np.sum(np.square(pts2 - Hpts1), axis=0))
        kept_index_list.append(np.where(dist < epsilon)[0])

    kept_index = kept_index_list[np.argmax(np.array([kept_index_list[i].shape[0] for i in range(n_test)]))]

    return matches[kept_index]


def ransac_matches_selection(kp1, kp2, matches, match_threshold=100):

    model = homography.RansacModel()

    pts1 = homography.make_homog(np.transpose(np.array([kp1[m.queryIdx].pt for m in matches])))
    pts2 = homography.make_homog(np.transpose(np.array([kp2[m.trainIdx].pt for m in matches])))

    H, inliers = homography.H_from_ransac(pts1, pts2, model, maxiter=1000, match_threshold=match_threshold)

    return matches[inliers]


def multiple_ransac_homography(kp1, kp2, matches, match_threshold=50, n_homog=5):
    model = homography.RansacModel()

    n_pts = len(matches)

    pts1 = homography.make_homog(np.transpose(np.array([kp1[m.queryIdx].pt for m in matches])))
    pts2 = homography.make_homog(np.transpose(np.array([kp2[m.trainIdx].pt for m in matches])))

    inliers_idx = np.array([],dtype=int)

    for i in range(n_homog):
        H, H_inliers = homography.H_from_ransac(pts1, pts2, model, maxiter=1000, match_threshold=match_threshold)
        inliers_idx = np.concatenate((inliers_idx,H_inliers))
        outliers_idx = np.array([(i not in inliers_idx) for i in range(n_pts)])

        pts1 = pts1[:,outliers_idx]
        pts2 = pts2[:,outliers_idx]
        n_pts = np.sum(outliers_idx)
    return matches[inliers_idx]