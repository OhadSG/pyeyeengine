import time

import cv2
import numpy as np

from pyeyeengine.calibration.auto_calibrator import AutoCalibrator
from pyeyeengine.camera_utils.camera_reader import CameraReader
from pyeyeengine.eye_engine.change_detector import ChangeDetector
from pyeyeengine.object_detection.ML_key_points_extractor import PointingFingerMLExtractor
from pyeyeengine.object_detection.object_detector import HandDetector
from pyeyeengine.tracking.tracker import Tracker, TrackedObject
from pyeyeengine.camera_utils.frame_manager import FrameManager
import pyeyeengine.utilities.global_params as Globals


class EyeEngine:
    def __init__(self, frame_grabber=None, background_model=ChangeDetector(),
                 object_detector=HandDetector(), key_pts_extractor=PointingFingerMLExtractor(),
                 tracker=Tracker(), calibrator=AutoCalibrator((320, 240))):
        # If you use CameraReader as default parameter then it is being
        # created when the module is imported, thus initializing the camera before we meant to use it
        FrameManager.getInstance().set_rgb_resolution(Globals.DEFAULT_DEPTH_IMAGE_SIZE)
        FrameManager.getInstance().set_depth_resolution(Globals.DEFAULT_DEPTH_IMAGE_SIZE)
        self._background_model = background_model
        self._object_detector = object_detector
        self._key_pts_extractor = key_pts_extractor
        self._tracker = tracker
        self._calibrator = calibrator
        self.is_run = True

    def set_key_points_extractor(self, key_pts_extractor):
        self._key_pts_extractor = key_pts_extractor

    def process_frame(self, display=False, show_time=False, record_fingers=False):
        if show_time:
            start_time = time.clock()

        if record_fingers or display:
            rbg = FrameManager.getInstance().get_rgb_frame()

        depth_map = FrameManager.getInstance().get_depth_frame()
        self._background_model.update_background_model(depth_map, self._object_detector.get_binary_objects())
        object_voxels = self._object_detector.process_frame(self._background_model.detect_change(depth_map), depth_map)
        object_voxels, key_pts_voxels = self._key_pts_extractor.extract(object_voxels, FrameManager.getInstance().get_rgb_frame())
        self._tracker.track(object_voxels, key_pts_voxels)
        transformed_objects = [TrackedObject(obj.id, self._calibrator.transfrom_points_cam_to_display(obj.key_pts))
                               for obj in self._tracker.get_tracked_objects()]

        if record_fingers:
            self.recrod_frame_data(rbg, self._background_model.num_of_frames_processed)

        if display:
            self.display_contours_on_rbg(self._object_detector.contours, key_pts_voxels, rbg, self._calibrator)

        if show_time:
            end_time = time.clock()
            print("engine run time (ms): %f" % ((end_time - start_time) * 1000))

        return transformed_objects

    def recrod_frame_data(self, img, key_pts_voxels, frame_num, vid_name):
        cv2.imwrite("finger_images/whole_frames/%s_%d.png" % (vid_name, frame_num), img)
        with open("finger_images/whole_frames/%s_%d.txt" % (vid_name, frame_num), "a") as f:
            for idx, key_pts in enumerate(key_pts_voxels):
                rect = [key_pts[0] - 1, key_pts[1] - 1, key_pts[0] + 1, key_pts[1] + 1]
                f.write("finger_images/%s_%d.txt engine_finger %d %d %d %d" %
                        (vid_name, frame_num, rect[0], rect[1], rect[2], rect[3]))

    def display_contours_on_rbg(self, contours, key_pts, img, calibrator):
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
        for key_pt in key_pts:
            img = cv2.circle(img, (np.int16(key_pt[0, 0]), np.int16(key_pt[0, 1])), 1, (0, 0, 255), -1)

        if calibrator:  # display edges of projected screen in rgb image
            top_right = np.expand_dims(calibrator.transfrom_points_display_to_cam(np.array([[0, 0]])), axis=1)
            top_left = np.expand_dims(
                calibrator.transfrom_points_display_to_cam(np.array([[calibrator.screen_width, 0]])), axis=1)
            bottom_right = np.expand_dims(
                calibrator.transfrom_points_display_to_cam(np.array([[0, calibrator.screen_height]])), axis=1)
            bottom_left = np.expand_dims(calibrator.transfrom_points_display_to_cam(
                np.array([[calibrator.screen_width, calibrator.screen_height]])), axis=1)
            cv2.line(img, (top_right[0, 0, 0], top_right[0, 0, 1]), (top_left[0, 0, 0], top_left[0, 0, 1]), (255, 0, 0),
                     5)
            cv2.line(img, (top_right[0, 0, 0], top_right[0, 0, 1]), (bottom_right[0, 0, 0], bottom_right[0, 0, 1]),
                     (255, 0, 0), 5)
            cv2.line(img, (bottom_right[0, 0, 0], bottom_right[0, 0, 1]), (bottom_left[0, 0, 0], bottom_left[0, 0, 1]),
                     (255, 0, 0), 5)
            cv2.line(img, (bottom_left[0, 0, 0], bottom_left[0, 0, 1]), (top_left[0, 0, 0], top_left[0, 0, 1]),
                     (255, 0, 0),
                     5)
        cv2.imshow('rbg with contours', img)
        cv2.waitKey(1)
