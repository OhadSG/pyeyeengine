import json
import time
import random
import numpy as np
import cv2
import os
import threading
from pyeyeengine.utilities.read_write_lock import RWLock
# from pyeyeengine.calibration.calibration_utilities import *
from pyeyeengine.utilities.helper_functions import *
import pyeyeengine.utilities.global_params as Globals
from pyeyeengine.utilities.logging import Log
from pyeyeengine.camera_utils.frame_manager import FrameManager
from pyeyeengine.eye_engine.fps_counter import FPSCounter
from pyeyeengine.eye_engine.change_detector import ChangeDetector
from pyeyeengine.calibration.ORB_single_image.ORB_single_image_calibration import SingleImageCalibrator
from pyeyeengine.object_detection.key_points_extractor import HandSilhouetteExtractor
from pyeyeengine.eye_engine.key_points_limiter import KeyPointsLimiter
from pyeyeengine.eye_engine.light_touch_engine import LightEyeEngine
from pyeyeengine.utilities.preferences import EnginePreferences

FLOW = "dynamic_surface_engine"
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
SAVED_FILES_FOLDER = FILE_PATH + "/../utilities/local_files/dynamic_surface_debug"
SAVED_FILES_FOLDER2 = "/root/dynamic_surface_debug"
# SAVED_FILES_FOLDER = FILE_PATH + "/../temp"
NORMALIZATION_LOWER_BOUND = -1.0
NORMALIZATION_UPPER_BOUND =  1.0

class DynamicSurfaceEngine():
    def __init__(self, frame_manager: FrameManager, calibrator=None, table_mask=None):
        Log.i("Dynamic Surface Engine Starting", flow=FLOW)
        os.system("rm -rf {}/*".format(SAVED_FILES_FOLDER))

        self._dynamic_surface_debug = EnginePreferences.getInstance().get_switch(EnginePreferences.DYNAMIC_SURFACE_DEBUG)
        self._far_plane = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.DYNAMIC_SURFACE_FAR_PLANE))
        self._near_plane = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.DYNAMIC_SURFACE_NEAR_PLANE))
        self._min_factor = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.DYNAMIC_SURFACE_MIN_FACTOR))
        self._change_threshold = string_to_float(EnginePreferences.getInstance().get_text(EnginePreferences.DYNAMIC_SURFACE_CHANGE_THRESHOLD))
        self.engine_type = Globals.EngineType.DynamicSurface
        self.table_mask = table_mask
        self.lock = RWLock()
        self._fps_counter = FPSCounter()
        self._background_model = ChangeDetector()
        self.__keypoints_engine = None
        self.__extractor = HandSilhouetteExtractor()
        self.__key_pts_limiter = KeyPointsLimiter(screen_width=1280, screen_height=800)
        self.should_sample_surface = False
        self._last_fetched_surface_data = {}
        self._surface_data_changed = False
        self.previous_sent_data = []
        self.previous_data = []
        self.switching_bool = True
        self.last_surface_request = None
        self.hand_min_height = 100 # the smallest distance a hand needs to be above a table to count as hand (in milimeters)
        self.table_mask_stretch_size = 30
        self.depth_stream = frame_manager.depth_stream

        self.depth_stream.set_resolution_named('small')
        self.processed_width = 640
        self.processed_height = 480
        if calibrator is None:
            self._calibrator = SingleImageCalibrator(frame_manager)
        else:
            self._calibrator = calibrator

        # Adjust table mask
        if self.table_mask is None or self.table_mask.shape[1] == 0:
            Log.e("Table mask not passed to DynamicSurface Engine")
            self.table_mask = np.ones((960, 1280, 3), 'uint8')

        cv2.imwrite(SAVED_FILES_FOLDER + "/dynamic_surface_table_mask.png", self.table_mask)
        self.table_mask = cv2.resize(self.table_mask, (Globals.DEPTH_HIGH_QUALITY.width, Globals.DEPTH_HIGH_QUALITY.height))
        self.table_mask = cv2.cvtColor(self.table_mask, cv2.COLOR_BGR2GRAY)

        # stretching the mask to remove blank spaces from the sides
        self.table_mask = cv2.resize(self.table_mask, (self.table_mask.shape[1] + self.table_mask_stretch_size * 2, self.table_mask.shape[0]), interpolation=cv2.INTER_AREA)
        self.table_mask = self.table_mask[:, self.table_mask_stretch_size:self.table_mask.shape[1] - self.table_mask_stretch_size]


    def process_frame(self):
        self.load_preferences()
        t1 = int(round(time.time() * 1000))
        surface_data = self.__sample_surface()
        t2 = int(round(time.time() * 1000))
        Log.d("time process_frame %s %s" % (t2 - t1, threading.current_thread().name))
        return surface_data

    def load_preferences(self):
        self._dynamic_surface_debug = EnginePreferences.getInstance().get_switch(EnginePreferences.DYNAMIC_SURFACE_DEBUG)
        self._far_plane = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.DYNAMIC_SURFACE_FAR_PLANE))
        self._near_plane = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.DYNAMIC_SURFACE_NEAR_PLANE))
        self._min_factor = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.DYNAMIC_SURFACE_MIN_FACTOR))
        self._change_threshold = string_to_float(EnginePreferences.getInstance().get_text(EnginePreferences.DYNAMIC_SURFACE_CHANGE_THRESHOLD))
        self.x_offset = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.X_OFFSET))
        self.y_offset = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.Y_OFFSET))
        self.x_scale = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.SCALE_X))
        self.y_scale = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.SCALE_Y))
        self.x_tilt = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.TILT))
        self.flip = True if EnginePreferences.getInstance().get_switch(EnginePreferences.FLIP) == "true" else False

    def __sample_surface(self):

        depth_data = self.depth_stream.get_frame()
        depth_data = cv2.resize(depth_data, (self.processed_width, self.processed_height),
                                interpolation = cv2.INTER_CUBIC)

        depth_data, modified_mask = self.apply_manual_calibration(depth_data)
        cropped_depth_data = crop_with_mask(depth_data, mask=modified_mask)

        cropped_depth_data = cv2.dilate(cv2.medianBlur(cv2.erode(cropped_depth_data, None, 3), 3), None, 5)

        # Remove black spots from borders
        temp_flattened_data = cropped_depth_data.flatten(order='C')
        flattened_data = temp_flattened_data.copy()
        value = self.median(temp_flattened_data)
        frame_width = 10
        modded_image = cropped_depth_data
        height, width = modded_image.shape
        modded_image[0:frame_width, :] = value
        modded_image[:, 0:frame_width] = value
        modded_image[height - frame_width:height, :] = value
        modded_image[:, width - frame_width:width] = value
        cropped_depth_data = modded_image

        height, width = cropped_depth_data.shape

        flattened_data, new_width, new_height = minimize_image(flattened_data, self._min_factor, width, height)
        minimum_depth_point = min(flattened_data)

        if minimum_depth_point == 0:
            minimum_depth_point = second_smallest(flattened_data)
        handled_data,hand_data = self.__handle_depth_data(flattened_data)

        if handled_data is None:
            handled_data = []
        else:
            self.previous_sent_data = handled_data

        hands = self.find_hands(hand_data, new_width, new_height)

        # Output
        json_object = {}
        json_object["width"] = new_width
        json_object["height"] = new_height
        json_object["values"] = handled_data
        json_object["hands"] = hands

        self._fps_counter.process_frame()
        Log.d("Surface data collected", flow=FLOW)
        return json_object

    def apply_manual_calibration(self, depth_data):
        if self.x_tilt != 0:
            gradient = np.tile(np.linspace(-self.x_tilt, self.x_tilt, depth_data.shape[1]).astype(int),
                               (depth_data.shape[0], 1))
            depth_data = np.add(depth_data, gradient).astype(np.int16)

        if self.x_scale != 0 or self.y_scale != 0:
            depth_data = cv2.resize(depth_data, (depth_data.shape[1] + self.x_scale * 2, depth_data.shape[0]),
                                    interpolation=cv2.INTER_AREA)
            if self.x_scale < 0:
                img = np.zeros((480, 640))
                img[:, -self.x_scale:img.shape[1] + self.x_scale] = depth_data
                depth_data = img.astype(np.int16)
                cv2.imwrite(FILE_PATH + "depth_data.png", depth_data)
            else:
                depth_data = depth_data[:, self.x_scale:depth_data.shape[1] - self.x_scale]

            depth_data = cv2.resize(depth_data, (depth_data.shape[1], depth_data.shape[0] + self.y_scale * 2),
                                    interpolation=cv2.INTER_AREA)
            if self.y_scale < 0:
                img = np.zeros((480, 640))
                img[-self.y_scale:img.shape[0] + self.y_scale:] = depth_data
                depth_data = img.astype(np.int16)
                cv2.imwrite(FILE_PATH + "depth_data.png", depth_data)
            else:
                depth_data = depth_data[self.y_scale:depth_data.shape[0] - self.y_scale, :]

        if self.flip:
            depth_data = np.flip(depth_data)

        modified_mask = np.roll(self.table_mask, [self.y_offset, self.x_offset], axis=(0, 1))

        return depth_data, modified_mask

    def find_hands(self, hand_data, new_width, new_height):
        # find hands only inside the of the image
        x_trim, y_trim = 5, 3
        cropped_hand_data = np.array(hand_data).reshape(new_height, new_width)[y_trim:-y_trim, x_trim:-x_trim]

        if 255 not in cropped_hand_data:
            return []

        ret, binary = cv2.threshold(cropped_hand_data.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hands = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
        if (w>5 and h>2)or (w>2 and h>5):
            hands.append({"x": math.floor(x + w / 2 + x_trim), "y": math.floor(y + h / 2 + y_trim)})
        return hands

    def __handle_depth_data(self, data):
        normalized, orig_data_fixed, hand_data = [],[],[]

        upper_lower_difference = float(NORMALIZATION_UPPER_BOUND - NORMALIZATION_LOWER_BOUND)
        max_min_difference = float(self._far_plane - self._near_plane)

        for i, data_point in enumerate(data):
            previous_frame_data_point = None
            above=0

            if len(self.previous_data) > 0 and self.previous_data[i]:
                previous_frame_data_point = self.previous_data[i]

            # Fix points that are above and below our minimum thresholds
            if data_point < self._near_plane:
                if data_point>0 and self._near_plane - data_point > self.hand_min_height:
                    above = 255
                if previous_frame_data_point is not None:
                    data_point = previous_frame_data_point
                else:
                    data_point = self._near_plane
            elif data_point > self._far_plane:
                data_point = self._far_plane

            # Save the fixed original data for reference in the next frame
            orig_data_fixed.append(data_point)

            # Normalize the data point
            data_point = upper_lower_difference * ((data_point - self._near_plane) / max_min_difference) + NORMALIZATION_LOWER_BOUND
            data_point = float("{:.3f}".format(data_point))
            hand_data.append(above)
            normalized.append(data_point)

        sum_of_orig_data = sum(orig_data_fixed)
        sum_of_prev_data = sum(self.previous_data)
        difference = abs(sum_of_prev_data - sum_of_orig_data)

        if difference < self._change_threshold:
            Log.d("Changes are minor, ignoring", extra_details={"difference": "{:.2f}".format(difference)})
            return None

        self.previous_data = orig_data_fixed
        return normalized,hand_data

    def median(self, lst):
        lst.sort()  # Sort the list first
        if len(lst) % 2 == 0:  # Checking if the length is even
            # Applying formula which is sum of middle two divided by 2
            return (lst[len(lst) // 2] + lst[(len(lst) - 1) // 2]) / 2
        else:
            # If length is odd then get middle value
            return lst[len(lst) // 2]

    def stop_surface_sampling(self):
        self.should_sample_surface = False






