import numpy as np


def fit_plane(voxels, iterations=50, inlier_thresh=10, num_pts_per_param=10, perc_validation=.1):  # voxels : x,y,z
    inliers, planes = [], []
    val_voxels = voxels[np.random.choice(voxels.shape[0], int(voxels.shape[0]*perc_validation)), :]
    xy1 = np.concatenate([val_voxels[:, :-1], np.ones((val_voxels.shape[0], 1))], axis=1)
    z = val_voxels[:,-1].reshape(-1,1)
    for _ in range(iterations):
        random_pts = voxels[np.random.choice(voxels.shape[0], voxels.shape[1] * num_pts_per_param, replace=False), :]
        plane_transformation, residual = fit_pts_to_plane(random_pts)
        inliers.append(((z - np.matmul(xy1, plane_transformation)) <= inlier_thresh).sum())
        planes.append(plane_transformation)
    best_plane = planes[np.array(inliers).argmax()]
    plane_transformation, residual = fit_pts_to_plane(
        val_voxels[((z - np.matmul(xy1, best_plane)) <= inlier_thresh*2).reshape(-1), :])
    return plane_transformation


def fit_pts_to_plane(voxels):  # x y z  (m x 3)
    # https: // math.stackexchange.com / questions / 99299 / best - fitting - plane - given - a - set - of - points
    xy1 = np.concatenate([voxels[:, :-1], np.ones((voxels.shape[0], 1))], axis=1)
    z = voxels[:, -1].reshape(-1, 1)
    fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(xy1.T, xy1)), xy1.T), z)
    errors = z - np.matmul(xy1, fit)
    residual = np.linalg.norm(errors)
    return fit, residual


# https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points

def planeFit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    import numpy as np
    from numpy.linalg import svd
    points = np.reshape(points, (points.shape[0], -1))  # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    center = points.mean(axis=1)
    x = points - center[:, np.newaxis]
    M = np.dot(x, x.T)  # Could also use np.cov(x) here.
    normal = svd(M)[0][:, -1]
    return center, normal


def get_plane_height_at(pts, plane_center, plane_normal):
    return (pts[:, 0] - plane_center[0]) * plane_normal[0] + (pts[:, 1] - plane_center[1]) * plane_normal[1]


def fit_line(pts, iterations=50, thresh=2):
    inliers, lines, mean_dist = [], [], []
    for _ in range(iterations):
        random_pts = pts[np.random.choice(pts.shape[0], pts.shape[1] * 10), :]
        line_pars, _, _, _, _ = np.polyfit(random_pts[:, 0], random_pts[:, 1], 1, full=True)
        line = np.poly1d(line_pars)
        if not np.isnan(line[0]):
            dist_pts_to_line = np.abs((line[1] * pts[:, 0] + line[0]) - pts[:, 1])
            inliers_curr = dist_pts_to_line <= thresh
            inliers.append(inliers_curr.sum())
            mean_dist.append(dist_pts_to_line[inliers_curr].mean())
            lines.append(line)
    if len(inliers) > 0 and np.array(inliers).max() > pts.shape[0] / 6:
        max_inliers = np.array(inliers).max()
        valid_lines = np.array(inliers) >= max_inliers*.85
        best_line = valid_lines * (np.array(mean_dist) == np.array(mean_dist)[valid_lines].min())
        return lines[best_line.argmax()], np.array(inliers).max() / pts.shape[0]
    else:
        return np.poly1d([np.nan, np.nan]), 0


def fit_line_weighted(pts, weights, iterations=50, thresh=2):
    inliers, lines, mean_dist = [], [], []
    for _ in range(iterations):
        random_indicies = np.random.choice(pts.shape[0], pts.shape[1] * 5)
        random_pts = pts[random_indicies, :]
        random_weights = weights[random_indicies]
        line_pars, _, _, _, _ = np.polyfit(random_pts[:, 0], random_pts[:, 1], 1, full=True, w=random_weights)
        line = np.poly1d(line_pars)
        if not np.isnan(line[0]):
            dist_pts_to_line = np.abs((line[1] * pts[:, 0] + line[0]) - pts[:, 1])
            inliers_curr = (dist_pts_to_line <= thresh)
            inliers.append((inliers_curr*weights).sum())
            mean_dist.append((dist_pts_to_line[inliers_curr]).mean())
            lines.append(line)
    if len(inliers) > 0 and np.array(inliers).max() > weights.sum() / 6:
        # max_inliers = np.array(inliers).max()
        # valid_lines = np.array(inliers) >= max_inliers*.85
        # best_line = valid_lines * (np.array(mean_dist) == np.array(mean_dist)[valid_lines].min())
        return lines[np.array(inliers).argmax()], np.array(inliers).max() / pts.shape[0]
    else:
        return np.poly1d([np.nan, np.nan]), 0


# ransac = linear_model.RANSACRegressor()
# ransac.fit(X, y)