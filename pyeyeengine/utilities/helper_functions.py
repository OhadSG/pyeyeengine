#import least-square functions to calculate the optimal plane

import numpy as np
import random
from scipy.optimize import leastsq
from pyeyeengine.utilities.astra_orbbec_conf import AstraOrbbec

DEBUG_MODE = False

def removeSublist(lst):
    curr_res = []
    result = []
    for ele in sorted(map(set, lst), key=len, reverse=True):
        if not any(ele <= req for req in curr_res):
            curr_res.append(ele)
            result.append(list(ele))

    return result

def f_min(points, p):
    plane_xyz = p[0:3]
    distance = (plane_xyz * points).sum(axis=1) + p[3]
    return distance / np.linalg.norm(plane_xyz)

def residuals(params, points):
    return f_min(points, params)

def calc_plane(points, guess):
    sol = leastsq(residuals, guess, args=(points), xtol=0.000001, ftol=1e-300)[0]
    # print((f_min(points, sol) ** 2).sum())
    # print(sol)

    return sol, (f_min(points, sol) ** 2).sum()

# Inital guess of the plane
p0 = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

def get_best_plane_from_points(points, guess=None):
    if guess is None:
        guess = p0

    plane, error = calc_plane(points, guess)
    norm = np.linalg.norm(plane[0:3])
    # normal = [plane[0] / norm, plane[1] / norm, plane[2] / norm]
    normal = [plane[0], plane[1], plane[2]]
    distance = plane[3]
    return normal, distance, error

# Filter by y-values of 3D-plane
def filter_by_function_3D(coord, points, margin=200):
    y = (coord[0] * points[:, 0] + coord[2] * points[:, 2] + coord[3]) / (-coord[1])
    valid_idx = np.where(points[:, 1] > y + margin, True, False)
    return points[valid_idx, :]

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def euclidean_to_pixels(points):
    result = np.zeros((len(points), 2))

    camera_conf = AstraOrbbec()
    result[:, 0] = np.uint16(np.round(-((points[:, 1] * camera_conf.f_x) / points[:, 2]) + camera_conf.c_x))  # row vector
    result[:, 1] = np.uint16(np.round(((points[:, 0] * camera_conf.f_y) / points[:, 2]) + camera_conf.c_y))  # col vector

    return result

def frame_image(image, channel):
    assert image is not None, "Image cannot be None!"
    assert channel >= 0 and channel <= 2, "Channel must be a value of RGB (0-2)"

    modded_image = image
    height, width, _ = modded_image.shape

    for index in range(3):
        value = 0

        if index == channel:
            value = 255

        modded_image[0:10, :, index] = value
        modded_image[:, 0:10, index] = value
        modded_image[height - 10:height, :, index] = value
        modded_image[:, width - 10:width, index] = value

    return modded_image