#import least-square functions to calculate the optimal plane

import numpy as np
import random
import math
from scipy.optimize import leastsq
from pyeyeengine.utilities.astra_orbbec_conf import AstraOrbbec

DEBUG_MODE = False

# Helper Classes

class Point():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def string(self):
        return "({},{})".format(self.x, self.y)

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
    assert channel >= 0 and channel <= 2, "Channel must be one of [R:0, G:1, B:2]"

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

def calculate_rectangle_area(top_left=Point(), bottom_right=Point()):
    height = bottom_right.y - top_left.y
    width = bottom_right.x - top_left.x
    return  height * width

# tests if angle abc is a right angle
def is_orthogonal(a=Point(), b=Point(), c=Point()):
    return (b.x - a.x) * (b.x - c.x) + (b.y - a.y) * (b.y - c.y) == 0

def is_rectangle(a=Point(), b=Point(), c=Point(), d=Point()):
    return is_orthogonal(a, b, c) and is_orthogonal(b, c, d) and is_orthogonal(c, d, a)

def is_rectangle_any_order(a=Point(), b=Point(), c=Point(), d=Point()):
    return is_rectangle(a, b, c, d) or is_rectangle(b, c, a, d) or is_rectangle(c, a, b, d)

def get_mask_corners(top_left=Point(), bottom_right=Point()):
    bottom_left = Point(top_left.x, bottom_right.y)
    top_right = Point(bottom_right.x, top_left.y)

    return top_left, bottom_left, top_right, bottom_right

def crop_with_mask(input_image, tolerance=0, mask=None):
    if mask is None:
        mask = input_image > tolerance
    try:
        return input_image[np.ix_(mask.any(1), mask.any(0))]
    except:
        print("here error1")
        return input_image

def second_smallest(lst):
    first = second = 5000
    for num in lst:
        if num < first:
            second, first = first, num
        elif first < num < second:
            second = num
    return second

def second_largest(lst):
    first = second = 0
    for num in lst:
        if num > first:
            second, first = first, num
        elif first > num > second:
            second = num
    return second

def trim_floats(input, decimal_places=2):
    if isinstance(input, float):
        return round(input, decimal_places)
    else:
        return [trim_floats(i, decimal_places) for i in input]

def normalize(data, lower_bound, upper_bound):
    min_data = min(data)
    max_data = max(data)
    normalized = [(upper_bound - lower_bound) * ((i - min_data) / (max_data - min_data)) + lower_bound for i in data]
    return normalized

def minimize_image(image_data, factor, original_width, original_height):
    if (original_width / factor) % 2 > 0:
        row_samples = int(math.ceil(original_width / factor))
        h_step = int(math.ceil(original_width / row_samples))
    else:
        row_samples = int(original_width / factor)
        h_step = int(original_width / row_samples)

    if (original_height / factor) % 2 > 0:
        col_samples = int(math.ceil(original_height / factor))
        v_step = int(math.ceil(original_height / col_samples))
    else:
        col_samples = int(original_height / factor)
        v_step = int(original_height / col_samples)

    i = j = 0
    minimized = []

    while j < original_height:
        i = 0
        while i < original_width:
            minimized.append(image_data[(j * original_width) + i])
            i += h_step
        j += v_step

    return minimized, row_samples, col_samples

def string_to_int(string):
    try:
        return int(string)
    except:
        return 0

def string_to_float(string):
    try:
        return float(string)
    except:
        return 0