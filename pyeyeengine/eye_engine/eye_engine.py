import time
# import matplotlib.pyplot as plt
import cv2
import numpy as np
import logging

from pyeyeengine.calibration.auto_calibrator import AutoCalibrator
from pyeyeengine.eye_engine.change_detector import ChangeDetector
from pyeyeengine.eye_engine.key_points_limiter import KeyPointsLimiter
from pyeyeengine.object_detection.key_points_extractor import PointingFingerExtractor
from pyeyeengine.object_detection.object_detector import HandDetector
from pyeyeengine.eye_engine.fps_counter import FPSCounter
from pyeyeengine.tracking.tracker import Tracker, TrackedObject
import pyeyeengine.utilities.global_params as Globals
from pyeyeengine.camera_utils.frame_manager import FrameManager
from .engine_base import EngineBase

logger = logging.getLogger(__name__)

class EyeEngine(EngineBase):
    engine_type = Globals.EngineType.EyeEngine

    def __init__(self,
        frame_manager: FrameManager,
        background_model=ChangeDetector(),
        object_detector=HandDetector(),
        key_pts_extractor=PointingFingerExtractor(),
        tracker=Tracker(),
        calibrator=None,
        fps_counter=FPSCounter(),
        key_pts_limiter=KeyPointsLimiter(screen_width=1280, screen_height=800)
    ):
        super().__init__()

        # If you use CameraReader as default parameter then it is being
        # created when the module is imported, thus initializing the camera before we meant to use it
        self.frame_manager = frame_manager
        self._background_model = background_model
        self._object_detector = object_detector
        self._key_pts_extractor = key_pts_extractor
        self._tracker = tracker
        self._calibrator = calibrator if calibrator is not None else AutoCalibrator(self.frame_manager, (640, 480),
                                                                                    camera=None)
        if hasattr(self._key_pts_extractor, 'max_height_above_playing_surface'):
            self._object_detector.set_max_height(self._key_pts_extractor.max_height_above_playing_surface)
        if hasattr(self._key_pts_extractor, "DIFF_NOISE_THRESHOLD"):
            self._background_model.DIFF_NOISE_THRESHOLD = self._key_pts_extractor.DIFF_NOISE_THRESHOLD
        self.is_run = True
        self.show_voxels = False
        self._fps_counter = fps_counter
        self._key_pts_limiter = key_pts_limiter

    def set_key_points_extractor(self, key_pts_extractor):
        self._key_pts_extractor = key_pts_extractor

    def process_frame(self, display=False, show_time=False, debug=False):
        if show_time:
            start_time = time.monotonic()

        depth_stream = self.frame_manager.depth_stream

        depth_stream.set_resolution(Globals.DEFAULT_CAMERA_RESOLUTION)
        # depth_map = cv2.resize(depth_stream.get_frame(), (320, 240))
        depth_map = depth_stream.get_frame()
        self._background_model.update_background_model(depth_map, self._object_detector.get_binary_objects(),
                                                       self._calibrator.table_mask)
        diff_map = self._background_model.detect_change(depth_map)
        object_voxels = self._object_detector.process_frame(diff_map, depth_map)
        objects_centroids, key_pts_voxels = self._key_pts_extractor.extract(object_voxels, None)
        objects_centroids, key_pts_voxels = self._object_detector.remove_irrelevant_objects(objects_centroids,
                                                                                            key_pts_voxels,
                                                                                            object_voxels, diff_map)

        self._tracker.track(objects_centroids, key_pts_voxels)

        transformed_objects = [TrackedObject(obj.id, self._key_pts_limiter.apply_limitations(
            self._calibrator.transfrom_points_cam_to_display(obj.get_key_points())))
                               for obj in self._tracker.get_tracked_objects()]

        self._fps_counter.process_frame()

        logger.debug('Process frame', extra={
            'fps': self._fps_counter.fps
        })

        if show_time:
            end_time = time.monotonic()
            print("engine run time (ms): %f" % ((end_time - start_time) * 1000))

        if display:
            FrameManager.getInstance().set_rgb_resolution(Globals.DEFAULT_CAMERA_RESOLUTION)
            rbg = cv2.resize(FrameManager.getInstance().get_rgb_frame(), (320, 240))
            self.display_contours_on_rbg(self._object_detector.contours, key_pts_voxels, rbg, self._calibrator)
        if debug:
            cv2.imshow("depth_map", depth_map * 5)
            cv2.imshow("binary_objects", np.uint8((self._object_detector.get_binary_objects() > 0) * 255))
            cv2.imshow("background_model", np.uint8((np.clip(np.float32(self._background_model.background_model),
                                                             a_min=500, a_max=3500) - 500) / 3000 * 255))

            cv2.imshow("diff_map", np.uint8(np.clip(np.float32(diff_map), a_min=-1000, a_max=1000) / 8 + 125))
            cv2.waitKey(5)

            # self._tracker.plot_tracker(rbg)
        return transformed_objects

    def is_tracked_object_click(self, tracked_objects):
        tracked_objects_out = []
        for obj in tracked_objects:
            velocity = np.diff(np.concatenate(obj.key_pts_history, axis=0), axis=0)
            obj.is_click = (velocity[-2, :] - velocity[-1, :]).max() > 3
            tracked_objects_out.append(obj)

        return tracked_objects_out

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
                     1)
            cv2.line(img, (top_right[0, 0, 0], top_right[0, 0, 1]), (bottom_right[0, 0, 0], bottom_right[0, 0, 1]),
                     (255, 0, 0), 1)
            cv2.line(img, (bottom_right[0, 0, 0], bottom_right[0, 0, 1]), (bottom_left[0, 0, 0], bottom_left[0, 0, 1]),
                     (255, 0, 0), 1)
            cv2.line(img, (bottom_left[0, 0, 0], bottom_left[0, 0, 1]), (top_left[0, 0, 0], top_left[0, 0, 1]),
                     (255, 0, 0),
                     1)
        cv2.imshow('rbg with contours', img)
        cv2.waitKey(1)
