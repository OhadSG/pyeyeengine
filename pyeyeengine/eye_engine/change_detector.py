import cv2
import numpy as np

from pyeyeengine.eye_engine.ransac_utils import fit_plane

BACKGROUND_LEARNING_RATE = .1
NUM_FRAMES_BETWEEN_UPDATE = 60
NUM_FRAMES_BEFORE_STABLE_BACKGROUND = 60
EXPECTED_FRAME_RATE = 30 / NUM_FRAMES_BETWEEN_UPDATE
SECONDS_TILL_DETECTION_IS_STALE = 30
STATIC_OBJECTS_THRESHOLD = .25
STATIC_OBJECTS_LEARNING_RATE = 1 / (EXPECTED_FRAME_RATE * SECONDS_TILL_DETECTION_IS_STALE) * STATIC_OBJECTS_THRESHOLD

MAXIMUM_OBJECT_COVERAGE = .2
MAX_FRAME_COUNT = 20000  # about 5 minutes
INITIAL_FAST_LEARN_NUM_FRAMES = 300
SIGNIFICANT_PART_OF_SCREEN = .03


class ChangeDetector:
    def __init__(self):
        self.background_model = np.zeros((240, 320))
        self.num_of_frames_processed = -INITIAL_FAST_LEARN_NUM_FRAMES  # at first run learn fast for 30 seconds.
        self.counter_since_object_found_map = np.zeros((240, 320))
        self.potential_static_objects = np.ones((240, 320), dtype=np.float32) * STATIC_OBJECTS_THRESHOLD
        self.very_noisy = False
        self.table_mask = np.ones((240, 320, 3)) * 255
        self.table_mask_dilated = np.ones((240, 320, 3)) * 255
        self.DIFF_NOISE_THRESHOLD = 25

    def detect_change(self, depth_map, mask=None):
        return np.where(self.table_mask_dilated[:,:,0] > 0,
                        cv2.subtract(np.uint16(self.background_model_adjusted), depth_map), 0)

    def update_background_model(self, depth_map, objects_binary_img, table_mask=np.ones((240, 320, 3)) * 255):

        if (self.num_of_frames_processed > NUM_FRAMES_BEFORE_STABLE_BACKGROUND) \
                and self.num_of_frames_processed % NUM_FRAMES_BETWEEN_UPDATE == 0 and not self.very_noisy:
            objects_binary_img = cv2.dilate(np.uint8(objects_binary_img * 255), None, iterations=3) > 0
            mask = self.create_mask_of_likely_background(objects_binary_img, depth_map)
            self.update_for_sure_static_objects(depth_map, objects_binary_img)
            cv2.accumulateWeighted(depth_map, self.background_model, BACKGROUND_LEARNING_RATE, mask)
            if table_mask.min() == 0:
                self.background_model_adjusted = self.adjust_depth_to_table_height(self.background_model, table_mask)
            else:
                self.background_model_adjusted = self.background_model

            self.very_noisy = self.check_if_noisy(depth_map)
            # print("slow_learning")

        elif (self.num_of_frames_processed < NUM_FRAMES_BEFORE_STABLE_BACKGROUND) or self.very_noisy:
            self.background_model = (1 - BACKGROUND_LEARNING_RATE) * self.background_model + \
                                    BACKGROUND_LEARNING_RATE * depth_map
            self.background_model_adjusted = self.background_model
            if self.num_of_frames_processed % NUM_FRAMES_BETWEEN_UPDATE == 0:
                self.very_noisy = self.check_if_noisy(depth_map)
            # print("fast_learning")

        elif self.num_of_frames_processed == NUM_FRAMES_BEFORE_STABLE_BACKGROUND:
            self.potential_static_objects = np.ones((240, 320), dtype=np.float32) * STATIC_OBJECTS_THRESHOLD
            self.very_noisy = False
            # print("refresh")
        else:
            # print("no_learning")
            pass

        self.num_of_frames_processed += 1
        self.num_of_frames_processed = self.num_of_frames_processed % MAX_FRAME_COUNT

    def check_if_noisy(self, depth_map):
        signigicant_portion_of_screen = (SIGNIFICANT_PART_OF_SCREEN * depth_map.shape[0] * depth_map.shape[1])
        diff = self.detect_change(depth_map)
        num_noisy_pixels = (cv2.inRange(np.abs(diff), self.DIFF_NOISE_THRESHOLD,
                                        self.DIFF_NOISE_THRESHOLD * 2) > 0).sum()
        binary_objects = cv2.inRange(diff, 15, 255) > 0
        # print("percent noisy ", num_noisy_pixels / (depth_map.shape[0] * depth_map.shape[1]))
        # print(binary_objects.mean())
        return num_noisy_pixels > signigicant_portion_of_screen or binary_objects.mean() > .2

    def update_for_sure_static_objects(self, depth_map, objects_binary_img):
        static_objects = self.potential_static_objects >= STATIC_OBJECTS_THRESHOLD

        diff = depth_map - self.background_model  # bigger number = closer to ground ( farther from camera )
        static_people_who_moved = np.logical_and((diff > 225), (depth_map > 0))

        eroded_background = cv2.erode(self.background_model, None, 1)
        depth_local_noise = np.abs(depth_map - eroded_background)
        noise_in_background_model = np.logical_and(
            np.logical_and((self.background_model > eroded_background),
                           cv2.inRange(depth_local_noise, 5, self.DIFF_NOISE_THRESHOLD)),
            (objects_binary_img == 0))
        to_reset_with_depth_map = np.logical_or(noise_in_background_model,
                                                np.logical_or(static_objects, static_people_who_moved))
        self.background_model = np.where(to_reset_with_depth_map, depth_map, self.background_model)
        self.potential_static_objects[static_objects] = 0

    def create_mask_of_likely_background(self, objects_binary_img, depth_map):
        if self.num_of_frames_processed > NUM_FRAMES_BEFORE_STABLE_BACKGROUND:
            diff = depth_map - self.background_model
            cv2.accumulateWeighted(np.float32(((diff < -225) + objects_binary_img) > 0),
                                   self.potential_static_objects, STATIC_OBJECTS_LEARNING_RATE)
            mask = np.uint8(diff >= -self.DIFF_NOISE_THRESHOLD) * 255  # update slowly everything below 25 ( likely noise )
        else:
            mask = np.ones_like(self.counter_since_object_found_map).astype(
                np.uint8)  # assume all is background for now.

        mask[depth_map == 0] = 0  # dont update where invalid depth
        mask[objects_binary_img] = 0  # don't update where there is an object
        return cv2.erode(mask, None, 3)  # filter noise

    def display(self, depth_map):
        diff = depth_map - self.background_model
        diff_normed = np.abs(diff) / 100
        cv2.imshow('depth diff', diff_normed)
        cv2.imshow('depth map normed', np.uint8(depth_map / self.background_model.max() * 255))
        cv2.imshow('background model', np.uint8(self.background_model / self.background_model.max() * 255))
        cv2.waitKey(1)
        cv2.imshow("static_obj_counter", np.uint8(self.potential_static_objects / STATIC_OBJECTS_THRESHOLD * 255))
        cv2.waitKey(1)

    def adjust_depth_to_table_height(self, depth_map, mask):
        self.update_table_masks(mask, depth_map)
        # mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        table_mask_bool = cv2.erode(self.table_mask[:, :, 0], None, 3, iterations=2) > 0
        voxels = np.concatenate([np.fliplr(np.array(np.where(table_mask_bool)).T),
                                 depth_map[table_mask_bool].reshape(-1, 1)], axis=1)
        plane_tform = fit_plane(voxels, iterations=5, inlier_thresh=5)
        x, y = np.meshgrid(np.linspace(0, 319, 320), np.linspace(0, 239, 240))
        plane_over_entire_depth_map = (
            np.matmul(
                np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), np.ones((y.shape[0] * y.shape[1], 1))], axis=1),
                plane_tform)).reshape(
            x.shape)
        # plane_over_entire_depth_map = cv2.dilate(depth_map, None, 3, iterations=3)
        to_update = (
                cv2.dilate(np.uint8(self.table_mask[:, :, 0] == 0), None, 3,
                           iterations=2) * np.uint8(
            self.table_mask_dilated[:, :, 0] > 0))  # * (depth_map > plane_over_entire_depth_map) > 0)
        # + cv2.dilate(np.uint8(depth_map == 0), None, 5)) > 0
        depth_map = np.where(to_update, plane_over_entire_depth_map, depth_map)
        return depth_map

    def get_median_height(self):
        return np.median(np.abs(self.background_model[np.abs(self.background_model) > 0]))

    def update_table_masks(self, table_mask, depth_map):
        table_mask = cv2.resize(table_mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        # self.table_mask = table_mask
        # self.table_mask_dilated = cv2.dilate(table_mask, None, iterations=20)
        if len(table_mask.shape) < 3:
            table_mask_tmp = np.stack((table_mask,) * 3, axis=-1)

        try:
            if not (self.table_mask == table_mask_tmp).all():
                self.table_mask = table_mask_tmp
                self.table_mask_dilated = cv2.dilate(table_mask_tmp, None, iterations=20)
        except:
            pass
    # def adjust_depth_to_table_height(self, depth_map, mask):
    #     self.update_table_masks(mask, depth_map)
    #     # mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    #     table_mask_bool = cv2.erode(self.table_mask[:, :, 0], None, 3, iterations=2) > 0
    #     voxels = np.concatenate([np.fliplr(np.array(np.where(table_mask_bool)).T),
    #                              depth_map[table_mask_bool].reshape(-1, 1)], axis=1)
    #     plane_tform = fit_plane(voxels, iterations=5, inlier_thresh=5)
    #     x, y = np.meshgrid(np.linspace(0, 319, 320), np.linspace(0, 239, 240))
    #     plane_over_entire_depth_map = (
    #         np.matmul(
    #             np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), np.ones((y.shape[0] * y.shape[1], 1))], axis=1),
    #             plane_tform)).reshape(
    #         x.shape)
    #     # plane_over_entire_depth_map = cv2.dilate(depth_map, None, 3, iterations=3)
    #     to_update = (
    #             cv2.dilate(np.uint8(self.table_mask[:, :, 0] == 0), None, 3,
    #                        iterations=2) * np.uint8(
    #         self.table_mask_dilated[:,:,0] > 0))  # * (depth_map > plane_over_entire_depth_map) > 0)
    #     # + cv2.dilate(np.uint8(depth_map == 0), None, 5)) > 0
    #     depth_map = np.where(to_update, plane_over_entire_depth_map, depth_map)
    #     return depth_map
    #
    # def get_median_height(self):
    #     return np.median(np.abs(self.background_model[np.abs(self.background_model) > 0]))
    #
    # def update_table_masks(self, table_mask, depth_map):
    #     table_mask = cv2.resize(table_mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    #     if not (self.table_mask == table_mask).all():
    #         self.table_mask = table_mask
    #         self.table_mask_dilated = cv2.dilate(table_mask, None, iterations=20)
