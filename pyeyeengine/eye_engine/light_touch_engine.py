import time

import cv2
import numpy as np

from pyeyeengine.calibration.auto_calibrator import AutoCalibrator
from pyeyeengine.eye_engine.change_detector import ChangeDetector
from pyeyeengine.eye_engine.key_points_limiter import KeyPointsLimiter
from pyeyeengine.object_detection.key_points_extractor import SilhouetteExtractor
from pyeyeengine.object_detection.object_detector import LightTouchDetector
from pyeyeengine.eye_engine.fps_counter import FPSCounter
from pyeyeengine.tracking.tracker import Tracker, TrackedObject
from pyeyeengine.utilities.global_params import EngineType
import pyeyeengine.utilities.global_params as Globals
from pyeyeengine.camera_utils.frame_manager import FrameManager
from .engine_base import EngineBase

class LightEyeEngine(EngineBase):
    engine_type = EngineType.LightTouch

    def __init__(self,
                 frame_manager: FrameManager,
                 background_model=ChangeDetector(),
                 tracker=Tracker(), calibrator=None, fps_counter=FPSCounter(),
                 key_pts_limiter=KeyPointsLimiter(screen_width=1280, screen_height=800),
                 key_pts_extractor=SilhouetteExtractor()):
        super().__init__()

        self.frame_manager = frame_manager
        self._background_model = background_model
        self._object_detector = LightTouchDetector()
        self._tracker = tracker
        self._calibrator = calibrator if calibrator is not None else AutoCalibrator(self.frame_manager, (320, 240))
        self._key_pts_extractor = key_pts_extractor
        if hasattr(self._key_pts_extractor, 'max_height_above_playing_surface'):
            self._object_detector.max_search_height = self._key_pts_extractor.max_height_above_playing_surface
        if hasattr(self._key_pts_extractor, "DIFF_NOISE_THRESHOLD"):
            self._background_model.DIFF_NOISE_THRESHOLD = self._key_pts_extractor.DIFF_NOISE_THRESHOLD

        self.is_run = True
        self._fps_counter = fps_counter
        self._key_pts_limiter = key_pts_limiter

    def process_frame(self, display=False, show_time=False, debug=False):
        if show_time:
            start_time = time.monotonic()

        self.frame_manager.depth_stream.set_resolution(Globals.Resolution(320, 240))
        depth_map = self.frame_manager.depth_stream.get_frame()
        self._background_model.update_background_model(depth_map, self._object_detector.get_binary_objects())
        diff_map = self._background_model.detect_change(depth_map)
        object_voxels = self._object_detector.process_frame(diff_map, depth_map)
        transformed_objects = [TrackedObject(idx, self._key_pts_limiter.apply_limitations(
            self._calibrator.transfrom_points_cam_to_display(key_pts[:, :2])))
                               for idx, key_pts in enumerate(object_voxels)]

        self._fps_counter.process_frame()

        if show_time:
            end_time = time.monotonic()
            print("engine run time (ms): %f" % ((end_time - start_time) * 1000))

        if display:
            self.frame_manager.rgb_stream.set_resolution(Globals.Resolution(320, 240))
            rbg = self.frame_manager.rgb_stream.get_frame()
            self.display_contours_on_rbg(self._object_detector.contours, object_voxels, rbg, self._calibrator)
        if debug:
            cv2.imshow("depth_map", depth_map*5)
            cv2.imshow("binary_objects", np.uint8((self._object_detector.get_binary_objects() > 0) * 255))
            cv2.imshow("background_model", np.uint8((np.clip(np.float32(self._background_model.background_model),
                                                   a_min=500, a_max=3500)-500)/3000*255))

            cv2.imshow("diff_map", np.uint8(np.clip(np.float32(diff_map), a_min=-1000, a_max=1000)/8 + 125))
            cv2.waitKey(5)
        return transformed_objects

    def display_contours_on_rbg(self, contours, key_pts_list, img, calibrator):
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
        for key_pts in key_pts_list:
            for key_pt_num in range(key_pts.shape[0]):
                img = cv2.circle(img, (np.int16(key_pts[key_pt_num, 0]),
                                       np.int16(key_pts[key_pt_num, 1])), 1, (0, 0, 255), -1)

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
        cv2.imshow('rbg with contours', img ) #cv2.resize(img, (0, 0), fx=2, fy=2))
        cv2.waitKey(1)

    def set_key_points_extractor(self, key_pts_extractor):
        self._key_pts_extractor = key_pts_extractor