import time

import cv2
import numpy as np


def depth_map_2_binary(depth_map_diff, max_height=255):
    bin_img = cv2.inRange(depth_map_diff, 15, max_height)  # -255, -5)
    return filter_noise(bin_img)


def filter_noise(binary_image):
    return cv2.dilate(cv2.medianBlur(cv2.erode(binary_image, None, 3), 3), None, 5)


class Detector:
    def __init__(self, max_height_above_playing_surface=150, max_search_height=255):
        self.max_height_above_playing_surface = max_height_above_playing_surface
        self.max_search_height = max_search_height

    def set_max_height(self, max_height):
        self.max_height_above_playing_surface = max_height

    def remove_irrelevant_objects(self, objects_centroids, key_pts_voxels, object_voxels, diff_map):

        centroids_out, key_pts_out = [], []
        for centroid, key_pts, contour_voxels in zip(objects_centroids, key_pts_voxels, object_voxels):
            if len(key_pts) == 1:
                dist_to_key_point = np.linalg.norm(contour_voxels[:, :-1] - key_pts, axis=1).reshape(-1)
                closest_pts_to_key_pts = np.int32(contour_voxels[dist_to_key_point.argsort()]
                                                  [:5, :-1].mean(axis=0, keepdims=True))
            else:
                closest_pts_to_key_pts = key_pts
            if np.all(np.abs(diff_map[(closest_pts_to_key_pts[:, 1], closest_pts_to_key_pts[:, 0])])
                      <
                      self.max_height_above_playing_surface):
                centroids_out.append(centroid)
                key_pts_out.append(key_pts)

        return centroids_out, key_pts_out


class HandDetector(Detector):
    def __init__(self):
        Detector.__init__(self)
        self._binary_objects = np.uint8(np.zeros((240, 320)))
        self.contours = []
        self.set_max_height(70)

    def process_frame(self, depth_map_diff, depth_map):
        # start_time = time.clock()
        self._binary_objects = depth_map_2_binary(depth_map_diff, self.max_search_height)
        _, self.contours, _ = cv2.findContours(self._binary_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.contours = [cnt for cnt in self.contours if self.is_valid_hand_by_area(cnt)]
        contours_as_voxels = [self.contour_to_voxels(depth_map, cnt) for cnt in self.contours]
        temp = [hand_voxels for hand_voxels in contours_as_voxels if self.is_valid_hand(hand_voxels)]
        # print("_object_detector (ms): %f" % ((time.clock() - start_time) * 1000))
        return temp

    def get_binary_objects(self):
        return self._binary_objects

    def is_valid_hand_by_area(self, contour):
        threshold_area = 30  # 100  # originally 20 ( extremely over sensitive )
        return cv2.contourArea(contour) > threshold_area

    def is_valid_hand(self, voxels):
        min_contour_pts = [4, 10, 20]
        min_valid_pts_ratio = [.6, .2]
        num_valid_pts = self.get_valid_voxels(voxels).shape[0]

        is_min_valid_voxel_validation = num_valid_pts > min_contour_pts[0]
        is_min_valid_voxel_ratio_validation_1 = self.is_valid_voxel_ratio(voxels, min_contour_pts[1],
                                                                          min_valid_pts_ratio[0], num_valid_pts)
        is_min_valid_voxel_ratio_validation_2 = self.is_valid_voxel_ratio(voxels, min_contour_pts[2],
                                                                          min_valid_pts_ratio[1], num_valid_pts)
        return (is_min_valid_voxel_validation or is_min_valid_voxel_ratio_validation_1 or
                is_min_valid_voxel_ratio_validation_2)

    def is_valid_voxel_ratio(self, voxels, min_counter_pt, min_valid_pt_ratio, num_valid_pts):
        valid_pts_ratio = num_valid_pts / voxels.shape[0]
        return (num_valid_pts < min_counter_pt and
                valid_pts_ratio > min_valid_pt_ratio)

    def contour_to_voxels(self, depth_map, contour):
        x, y = contour[:, 0, 0], contour[:, 0, 1]
        return np.vstack((x, y, depth_map[y, x])).T

    def get_valid_voxels(self, voxels):
        return voxels[voxels[:, -1] > 0, :]


class LightTouchDetector(Detector):
    def __init__(self):
        Detector.__init__(self)
        self._binary_objects = np.uint8(np.zeros((240, 320)))
        self._binary_objects_tiny_prev = np.uint8(np.zeros((240, 320)))

    def process_frame(self, depth_map_diff, depth_map=None, factor=5):
        factor_inv = np.maximum(1 / factor, .05)
        self._binary_objects = depth_map_2_binary(depth_map_diff, self.max_search_height)
        binary_objs_tiny = np.uint8(cv2.resize(self._binary_objects, (0, 0), fx=factor_inv, fy=factor_inv) > 0) * 255
        binary_objs_tiny_curr = self.predict_next_bin_objs(self._binary_objects_tiny_prev, binary_objs_tiny)
        self._binary_objects_tiny_prev = binary_objs_tiny
        binary_objs_tiny = binary_objs_tiny_curr
        _, self.contours, _ = cv2.findContours(binary_objs_tiny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return [self.contour_to_voxels(cnt, factor) for cnt in self.contours if self.is_valid(cnt)]

    def get_binary_objects(self):
        return self._binary_objects

    def is_valid(self, contour):
        return (25 > contour.shape[0] > 2)  # and ((contour.mean(keep_dims=True)-contour).mean() > 2)

    def contour_to_voxels(self, contour, factor):
        x, y = contour[:, 0, 0], contour[:, 0, 1]
        return np.int32(np.vstack((x * factor + factor / 2, y * factor + factor / 2, np.ones_like(x))).T)

    def predict_next_bin_objs(self, prev_bin_objs, bin_objs):
        if prev_bin_objs.shape[0] == bin_objs.shape[0]:
            ret, labels_prev = cv2.connectedComponents(prev_bin_objs)
            ret, labels_curr = cv2.connectedComponents(bin_objs)
            out_bin_obj = np.zeros_like(bin_objs)
            for label_num in range(1, labels_curr.max()):
                overlaping_labels = labels_prev[labels_curr == label_num]
                if (overlaping_labels > 0).mean() > .25:
                    overlaping_labels = overlaping_labels[overlaping_labels > 0].tolist()
                    prev_label_match = \
                        max(map(lambda val: (overlaping_labels.count(val), val), set(overlaping_labels)))[1]
                    curr_obj_pts = np.stack(np.nonzero(labels_curr == label_num), axis=1)
                    centroid_diff = np.stack(np.nonzero(labels_prev == prev_label_match), axis=1).mean(axis=0) - \
                                    curr_obj_pts.mean(axis=0)
                    out_bin_obj[
                        np.clip(curr_obj_pts[:, 0] - int(1.5 * centroid_diff[0]), a_min=0,
                                a_max=out_bin_obj.shape[0] - 1),
                        np.clip(curr_obj_pts[:, 1] - int(1.5 * centroid_diff[1]), a_min=0,
                                a_max=out_bin_obj.shape[1] - 1)] = 1
            improved_bin_objs = np.uint8(((out_bin_obj + (bin_objs > 0)) > 0) * 255)
            # cv2.imshow("improved_bin_objs",
            #            cv2.resize(
            #                np.stack([np.uint8((out_bin_obj > 0) * 255), improved_bin_objs, bin_objs], axis=2)
            #                , (0, 0), fx=5, fy=5))
            # cv2.waitKey(1)
            return improved_bin_objs
        else:
            return bin_objs
