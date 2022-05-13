import time

import numpy as np
# from scipy.spatial.distance import cdist


def find_hand_base(voxels):
    valid_voxels = voxels[voxels[:, -1] > 0, :]
    bbox_X, bbox_X, bbox_W, bbox_H = calc_voxels_bounding_box(voxels)  # x,y,w,h
    # min_dist_to_surface = valid_voxels[:, -1].min()
    # center_of_mass = get_center_xy(voxels)
    hand_base_radius = max(bbox_W, bbox_H) / 2
    highest_pt = valid_voxels[np.argmin(valid_voxels[:, -1]), :-1]
    distances_2_highest_pt = dist_sq(valid_voxels[:, :-1], highest_pt)
    return valid_voxels[distances_2_highest_pt < hand_base_radius, :-1].mean(axis=0).reshape(1, 2)


def dist_sq(pts, pt):
    return ((pts - pt) ** 2).sum(axis=1)


def find_pointing_finger(voxels):
    hand_base_pt, pointing_finger_base = find_pointing_finger_base(voxels)
    return extend_finger(pointing_finger_base, hand_base_pt)


def find_pointing_finger_base(voxels):
    hand_base_pt = find_hand_base(voxels)
    dist_from_hand_base = dist_sq(voxels[:, :-1], hand_base_pt)
    pointing_finger_base = voxels[np.argmax(dist_from_hand_base), :-1].reshape(1, 2)
    return hand_base_pt, pointing_finger_base


def extend_finger(finger_pt, hand_base_pt):
    finger_dir_vector = finger_pt-hand_base_pt
    return finger_pt + (finger_dir_vector/np.linalg.norm(finger_dir_vector))*3


def find_foot_edges(voxels, radius_from_toe=5):
    _, toe_point = find_pointing_finger_base(voxels)
    dist_from_toe = dist_sq(voxels[:, :-1], toe_point.reshape(1, 2))
    return voxels[dist_from_toe < radius_from_toe, :-1].mean(axis=0).reshape(1, 2)


def calc_voxels_bounding_box(voxels):
    return [voxels[:, 0].min(), voxels[:, 1].min(), voxels[:, 0].ptp(), voxels[:, 1].ptp()]


def get_center_xy(voxels):
    return np.average(voxels[:, :-1], axis=0).reshape(1, -1)


def get_centroids(voxels_list):
    return [get_center_xy(voxels) for voxels in voxels_list]


class PointingFingerExtractor:
    def __init__(self):
        self.max_height_above_playing_surface = 50
        self.DIFF_NOISE_THRESHOLD = 15

    def extract(self, voxels_list, frame_grabber=None):
        return get_centroids(voxels_list), \
               [find_pointing_finger(voxels) for voxels in voxels_list]


class CentroidExtractor:
    def extract(self, voxels_list, frame_grabber=None):
        return get_centroids(voxels_list), get_centroids(voxels_list)


class SilhouetteExtractor:
    def __init__(self):
        self.max_height_above_playing_surface = 255
        self.DIFF_NOISE_THRESHOLD = 25

    def extract(self, voxels_list, frame_grabber=None):
        return get_centroids(voxels_list), \
               [np.unique(np.floor(voxels[:, :-1] / 10), axis=0) * 10 + 5
                for voxels in voxels_list]

class HandSilhouetteExtractor:
    def __init__(self):
        self.max_height_above_playing_surface = 70
        self.DIFF_NOISE_THRESHOLD = 15

    def extract(self, voxels_list, frame_grabber=None):
        return get_centroids(voxels_list), \
               [np.unique(np.floor(voxels[:, :-1] / 10), axis=0) * 10 + 5
                for voxels in voxels_list]



class FootEdgeExtractor:
    def __init__(self):
        self.DIFF_NOISE_THRESHOLD = 25

    def extract(self, voxels_list, frame_grabber=None):
        return get_centroids(voxels_list), [find_foot_edges(voxels) for voxels in voxels_list]


class RandomExtractor:
    def extract(self, voxel_list=None, frame_grabber=None):
        random_voxels = [np.hstack([np.random.randint(0, 320, (1, 1)), np.random.randint(0, 240, (1, 1))])
                         for _ in range(np.random.randint(0, 20))]
        return random_voxels, random_voxels


