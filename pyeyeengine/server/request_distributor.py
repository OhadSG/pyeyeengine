import base64
import os
import time
from pyeyeengine.utilities import global_params

import cv2
import json
import traceback
import threading

import numpy as np
from pyeyeengine.calibration.auto_calibrator import AutoCalibrator
from pyeyeengine.calibration.ORB_single_image.ORB_single_image_calibration import SingleImageCalibrator
from pyeyeengine.calibration.export_calibration import export_data_for_android
from pyeyeengine.eye_engine.eye_engine import EyeEngine
from pyeyeengine.eye_engine.fps_counter import FPSCounter
from pyeyeengine.eye_engine.key_points_limiter import KeyPointsLimiter
from pyeyeengine.eye_engine.light_touch_engine import LightEyeEngine
from pyeyeengine.center_of_mass.Engines.com_engine_ceiling import COMEngine
from pyeyeengine.object_detection.key_points_extractor import FootEdgeExtractor, PointingFingerExtractor, \
    SilhouetteExtractor, RandomExtractor, HandSilhouetteExtractor
from pyeyeengine.projector_controller.projector_controller import ProjectorController
from pyeyeengine.server.calibration_requests import _calibrate, _get_calibration_images_grid
from pyeyeengine.server.erorr_logger import ErrorLogger
from pyeyeengine.server.key_points_extractor_socket import KeyPointsExtractorSocket
from pyeyeengine.auto_keystones.auto_keystones_utils import fix_keystones, fix_keystones_using_epsons_autokeystones
import pyeyeengine.server.images_server as images_server
from pyeyeengine.utilities.logging import Log
from pyeyeengine.camera_utils.frame_manager import FrameManager
from pyeyeengine.utilities.file_uploader import FileUploader

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_ERROR_LOG = FILE_PATH + "/../error_log.txt"



class RequestDistributor:

    def __init__(self, screen_resolution=(1280, 800)) -> None:
        super().__init__()
        self._error_loger = ErrorLogger()
        self._engine = EyeEngine(key_pts_extractor=FootEdgeExtractor(),
                                 calibrator=AutoCalibrator(screen_resolution=screen_resolution),
                                 fps_counter=FPSCounter())
        self._camera_in_use = False
        self.screen_setter = images_server.screen_setter
        self.screen_setter.set_screen_res(screen_width=screen_resolution[0], screen_height=screen_resolution[1])
        self.single_image_calibrator = None
        self.engine_already_ran = False
        self._socket = KeyPointsExtractorSocket()
        FrameManager.getInstance().start()
        threading.Thread(target=self.prepare_projector).start()

    def prepare_projector(self):
        # Change driver permissions
        driver_path = FILE_PATH + "/../projector_controller/epscom-driver/"
        os.system("chmod -R 755 {}".format(driver_path))

        projector_controller = ProjectorController()

        if not projector_controller.is_aspect_full():
            projector_controller.set_aspect_to_full()
        if not projector_controller.is_dynamic_color_mode():
            projector_controller.set_dynamic_color_mode()

    def distribute(self, request_json):
        try:
            return json.dumps({'status': 'ok', 'data': self._try_serving(request_json)})
        except Exception as e:
            Log.e("Exception Caught", extra_details={"exception": "{}".format(e), "stacktrace": traceback.format_exc()})
            FrameManager.reset(reason="RequestDistributor Exception")
            with open(PATH_TO_ERROR_LOG, 'w') as outfile:
                traceback.print_exc(file=outfile)
            self._error_loger.add_error(str(e))
            return json.dumps({'status': 'failed', 'message': str(e)})

    def _try_serving(self, request):
        received_message_text = "Incoming request: '{}'".format(request["name"])
        if 'params' in request.keys():
            Log.i(received_message_text, extra_details=request["params"])
        else:
            Log.i(received_message_text)

        request_name = request["name"]

        if request_name == "test_request":
            FrameManager.getInstance().set_depth_resolution(global_params.Resolution(320, 240))
            depth_map = FrameManager.getInstance().get_depth_frame()
            return "ok"

        if request_name == "ping":
            return "Engine is responding"

        if request_name == "record":
            if request["params"]["record"] == 1:
                FrameManager.getInstance().start_recording()
            else:
                FrameManager.getInstance().stop_recording()
            return "ok"

        if request_name == "get_depth_frame":
            FrameManager.getInstance().set_depth_resolution(global_params.DEPTH_HIGH_QUALITY)
            FileUploader.upload_image(FrameManager.getInstance().get_depth_frame(), "requested_depth_frame.png")
            return "ok"

        if request_name == "get_error_log":
            return self._error_loger.get_log()

        if request_name == "show_image_in_server":
            if self.screen_setter.image.sum() == 0:
                self.screen_setter.set_image(np.ones_like(self.screen_setter.image) * 123)
            else:
                self.screen_setter.set_image(np.zeros_like(self.screen_setter.image))
            return "ok"

        if request_name == "reset_image_server":
            images_server.reset_thread()
            self.screen_setter.set_screen_res(screen_width=1280, screen_height=800)
            return "reset image server"

        if request_name == "reset_frame_manager":
            FrameManager.reset(reason="Client Requested")
            return "ok"

        if request_name == "reset_server":
            return self._reset_server()

        if request_name == "get_key_points":
            if request['params']['key_points_extractor'] == "pointing_finger":
                FileUploader.upload_image(FrameManager.getInstance().get_rgb_frame(), "camera_current_mask.png")
            return self._get_key_points(request)

        if request_name == "get_template_image":
            return self._bytes_to_base64(SingleImageCalibrator.get_template_image())

        if request_name == "get_rbg_as_png":
            return  self._bytes_to_base64(FrameManager.getInstance().get_rgb_image())

        if request_name == "get_grid_image_by_path":
            return self._bytes_to_base64(self._camera._camera.get_saved_img_as_png(request["params"]["image_path"]))

        if request_name == "save_rbg_as_png":
            return self._save_rgb_as_png(request)

        if request_name == "reload_table_data":
            return self._engine._calibrator.load_table_data()

        if request_name == "get_calibration_images_grid":
            return _get_calibration_images_grid(request)

        if request_name == "calibrate":
            _calibrate(self._engine._calibrator)
            return "calibrate done"

        if request_name == "calibrate_with_image_server":
            start_time = time.clock()
            if self._socket.is_active:
                self._socket.stop()
            screen_resolution = (request["params"]["screen_width"], request["params"]["screen_height"])
            self._engine._calibrator = AutoCalibrator(screen_resolution=screen_resolution)
            self._engine._calibrator.calibrate(mode=request["params"]["mode"],
                                               screen_setter=self.screen_setter)
            print("calibrate: %f (s)" % (time.clock() - start_time))
            return "calibrated sucessfully" if self._engine._calibrator.calibrate_success else "calibrated failed"

        if request_name == "calibrate_with_single_image":
            start_time = time.clock()

            if self._socket.is_active:
                self._socket.stop()

            screen_resolution = global_params.Resolution(request["params"]["screen_width"],
                                                         request["params"]["screen_height"])

            if self.single_image_calibrator is None:
                self.single_image_calibrator = SingleImageCalibrator(screen_resolution=global_params.SCREEN_RESOLUTION)

            self.single_image_calibrator.screen_resolution = screen_resolution

            self._engine._calibrator = self.single_image_calibrator
            self._engine._calibrator.calibrate(mode=request["params"]["mode"])

            Log.i("Calibration duration", extra_details={"duration": "{:.1f}".format(time.clock() - start_time)})
            return "calibrated sucessfully" if self._engine._calibrator.calibrate_success else "calibrated failed"

        if request_name == "reset_saved_data":
            return self.reset_saved_data()

        if request_name == "find_table":
            return self.find_table()

        if request_name == 'get_center_of_mass':
            return self._get_center_of_mass(request)

        if request_name == "get_white_background":
            return self._bytes_to_base64(cv2.imencode(".png",
                                                      self._engine._calibrator.generate_blurry_white_screen()
                                                      )[1].tobytes())

        if request_name == "set_screen_res":
            self._engine._calibrator.screen_width = request["params"]["screen_width"]
            self._engine._calibrator.screen_height = request["params"]["screen_height"]
            return "success"

        if request_name == "reset_table_mask":
            Log.i("Resetting Table Mask")
            path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(path, "../calibration")
            if os.path.exists(path + "/table_data_manual.npz"):
                os.remove(path + "/table_data_manual.npz")

            return "success"

        if request_name == "get_table_mask":
            path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(path, "../calibration")

            self._engine._calibrator.load_table_data()
            if os.path.exists(path + "/table_data_manual.npz"):
                table_mask = self._engine._calibrator.table_mask
                shape = self._engine._calibrator.table_shape
                edges = self._engine._calibrator.mask_edges.item()
                table_mask_with_alpha = np.concatenate([np.tile(table_mask[:, :, np.newaxis], (1, 1, 3)),
                                                        255 - table_mask[:, :, np.newaxis]], axis=2)
                return {
                    "image": self._bytes_to_base64(cv2.imencode(".png", table_mask_with_alpha)[1].tobytes()),
                    "top": "%d" % (edges['top']),
                    "left": "%d" % (edges['left']),
                    "bottom": "%d" % (edges['bottom']),
                    "right": "%d" % (edges['right']),
                    "shape": shape}

            table_mask, rect_display = export_data_for_android(self._engine._calibrator)
            table_mask_with_alpha = np.concatenate([np.tile(table_mask[:, :, np.newaxis], (1, 1, 3)),
                                                    255 - table_mask[:, :, np.newaxis]], axis=2)
            return {
                "image": self._bytes_to_base64(cv2.imencode(".png", table_mask_with_alpha)[1].tobytes()),
                "top": "%d" % (rect_display[1]),
                "left": "%d" % (rect_display[0]),
                "bottom": "%d" % (rect_display[3]),
                "right": "%d" % (rect_display[2]),
                "shape": "" if self._engine._calibrator.table_shape is None else self._engine._calibrator.table_shape}

        if request_name == "set_table_mask":
            print('params:', request['params'])
            if request["params"]["shape"]=='circle':
                center = [request["params"]["center_y"], request["params"]["center_x"]]
                radius = request["params"]["radius"]

                h = global_params.PROJECTOR_RESOLUTION.height
                w = global_params.PROJECTOR_RESOLUTION.width

                indices = self.create_circular_mask(h, w, center=center, radius=radius)
                img1 = np.zeros((h, w))
                img1[indices] = 255
                table_mask = np.uint8(img1)

                path = os.path.dirname(os.path.realpath(__file__))
                path = os.path.join(path, "../calibration")

                Log.i("Setting Manual Table Mask", extra_details={"shape":"circle",
                                                                  "center":"{}".format(center),
                                                                  "radius":"{}".format(radius)})
                np.savez(path + "/table_data_manual", table_shape='circle',
                         table_mask=table_mask, edge_params={"top":center[0] - radius,
                                                             "bottom":center[0] + radius,
                                                             "left":center[1] - radius,
                                                             "right":center[1] + radius})

            elif request["params"]["shape"]=='rectangle':
                h = global_params.PROJECTOR_RESOLUTION.height
                w = global_params.PROJECTOR_RESOLUTION.width

                top_left = [int(request["params"]["top_left_x"]), int(request["params"]["top_left_y"])]
                top_right = [int(request["params"]["top_right_x"]), int(request["params"]["top_right_y"])]
                bottom_right = [int(request["params"]["bottom_right_x"]), int(request["params"]["bottom_right_y"])]
                bottom_left = [int(request["params"]["bottom_left_x"]), int(request["params"]["bottom_left_y"])]

                contours = np.array([list(bottom_left), list(bottom_right), list(top_right), list(top_left)])
                img = np.zeros((h, w))
                cv2.fillPoly(img, pts=[contours], color=(255, 255, 255))
                table_mask = np.array(img, dtype='uint8')

                path = os.path.dirname(os.path.realpath(__file__))
                path = os.path.join(path, "../calibration")

                Log.i("Setting Manual Table Mask", extra_details={"shape":"rectangle",
                                                                  "top_left": "{}".format(top_left),
                                                                  "top_right": "{}".format(top_right),
                                                                  "bottom_right": "{}".format(bottom_right),
                                                                  "bottom_left": "{}".format(bottom_left)})
                np.savez(path + "/table_data_manual", table_shape='rectangle',
                         table_mask=table_mask, edge_params={"top": min(top_left[1], top_right[1]),
                                                             "bottom": max(bottom_left[1], bottom_right[1]),
                                                             "left": min(top_left[0], bottom_left[0]),
                                                             "right": max(top_right[0], bottom_right[0])})
            else:
                print('Wrong shape specified')

            return 1

        if request_name == "fix_keystones":
            self._fix_keystones(request, "beam")
            return ""

        if request_name == "fix_keystones_obie":
            self._fix_keystones(request, "obie")
            return ""

        if request_name == "get_monitor_data":
            return self._get_monitor_data()

        if request_name == "reset_keystones":
            self._set_default_keystones()
            return ""

        if request_name == "is_projector_turned_on":
            return ProjectorController().is_turned_on()

        if request_name == "turn_on_projector":
            return ProjectorController().turn_on()

        if request_name == "turn_off_projector":
            return ProjectorController().turn_off()

        if request_name == "get_projector_temp":
            return ProjectorController().get_temp()

        if request_name == "get_projector_lamp_hours":
            return ProjectorController().get_lamp_hours()

        if request_name == "change_projector_keystones":
            return ProjectorController().change_vertical_keystone(request["params"]["degrees"])

        if request_name == "get_fps":
            return "%f" % self._engine._fps_counter.fps

        if request_name == "take_high_res_photos":
            for _ in range(5):
                FrameManager.getInstance().get_rgb_image()

        raise ClientServerException("Unknown Command " + request_name)

    def _report_installation_issues(self):
        return {"noisy_area": self._engine._background_model.very_noisy,
                "camera_sees_whole_screen": self._engine._calibrator.can_camera_see_whole_screen(),
                "good_height_for_table": 4250 > self._engine._background_model.get_median_height() > 2500,
                "good_height_for_floor": 2250 > self._engine._background_model.get_median_height() > 1400}

    def _get_monitor_data(self):
        BASE_PATH = os.path.dirname(os.path.realpath(__file__))
        FULL_VERSION_FILE_NAME = "full_version.txt"
        FULL_VERSION_FILE_PATH = BASE_PATH + "/../" + FULL_VERSION_FILE_NAME

        version = "1.0.6.1111"

        with open(FULL_VERSION_FILE_PATH, "r") as file:
            version = file.readline()

        return { "version": version,
                "camera_serial_number": FrameManager.getInstance().get_serial_number() }

    def reset_saved_data(self):
        Log.i("Resetting Saved Data")
        BASE_PATH = os.path.dirname(os.path.realpath(__file__))

        HOMOGRAPHY = BASE_PATH + "/../calibration/warp_mat_cam_2_displayed.npy"
        if os.path.isfile(HOMOGRAPHY):
            os.remove(HOMOGRAPHY)

        CALIBRATION_PLANE = BASE_PATH + "/../calibration/calibrated_plane.npy"
        if os.path.isfile(CALIBRATION_PLANE):
            os.remove(CALIBRATION_PLANE)

        CALIBRATION_MASK = BASE_PATH + "/../calibration/calibrated_mask.npy"
        if os.path.isfile(CALIBRATION_MASK):
            os.remove(CALIBRATION_MASK)

        AUTO_TABLE_MASK = BASE_PATH + "/../calibration/table_data.npz"
        if os.path.isfile(AUTO_TABLE_MASK):
            os.remove(AUTO_TABLE_MASK)

        MANUAL_TABLE_MASK = BASE_PATH + "/../calibration/table_data_manual.npz"
        if os.path.isfile(MANUAL_TABLE_MASK):
            os.remove(MANUAL_TABLE_MASK)

        return "success"

    def find_table(self):
        self._engine._calibrator.try_to_find_table()
        return "sucess"

    def _save_rgb_as_png(self, request):
        time.sleep(.1)
        FrameManager.getInstance().save_rgb_as_png(request["params"]["save_path"])
        cv2.waitKey(1)
        return "success"

    def create_circular_mask(self, h, w, center=None, radius=None):
        if center is None:  # use the middle of the image
            center = [int(w / 2), int(h / 2)]
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        X, Y = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    def _reset_server(self):
        FrameManager.getInstance().reset(reason="Server Reset")
        self._engine = EyeEngine(key_pts_extractor=FootEdgeExtractor(),
                                 calibrator=AutoCalibrator(screen_resolution=(0, 0)),
                                 fps_counter=FPSCounter())
        return "success"

    def _get_center_of_mass(self, request):

        if not self._socket.is_active:
            self.init_center_of_mass()
            self._socket.set_engine(self._engine)
            self._socket.add_new_client(request['client'])
            ret_val = self._socket.start()
        else:
            ret_val = self._socket.add_new_client(request['client'])

        self.engine_already_ran = True
        return ret_val

    def init_center_of_mass(self):
        self._engine = COMEngine()
        self._engine.define_bg()
        Log.i("Center of mass engine initialized")

    def _get_key_points(self, request):
        if self.single_image_calibrator is None:
            self.single_image_calibrator = SingleImageCalibrator(screen_resolution=global_params.SCREEN_RESOLUTION)

        Log.d("Engine Status", extra_details={"engine_ran": "{}".format(self.engine_already_ran),
                                              "engine_fps": "{}".format(int(self._engine._fps_counter.fps)),
                                              "frame_manager_fps": "{}".format(FrameManager.getInstance().get_fps())})

        if (self.engine_already_ran and self._engine._fps_counter.fps < 10):
            Log.w("Engine FPS Dropped",
                  extra_details={"engine_fps": "{}".format(self._engine._fps_counter.fps)})
            self._socket.stop()

        if FrameManager.getInstance().get_fps() < 10:
            Log.w("Camera FPS Dropped",
                  extra_details={"frame_manager_fps": "{}".format(FrameManager.getInstance().current_fps)})
            FrameManager.reset(reason="Camera FPS Dropped")

        if not self._socket.is_active:
            Log.d("Using a new engine")
            self.init_engine_for_get_key_points(request)
            self._socket.set_engine(self._engine)
            self._socket.add_new_client(request['client'])
            ret_val = self._socket.start()
        else:
            Log.d("Using an existing engine")
            ret_val = self._socket.add_new_client(request['client'])

        self.engine_already_ran = True
        return ret_val

    def init_engine_for_get_key_points(self, request):
        background_model, calib = self.get_engine_state()
        key_pts_limiter = KeyPointsLimiter(screen_width=1280,  # request['params'].get('screen_width', 1280),
                                           screen_height=800)  # request['params'].get('screen_height', 800))

        extractor, engine_type = self._get_relevant_extractor_engine_type(request['params']['key_points_extractor'])
        if engine_type == "EyeEngine":
            Log.i("Starting Engine", extra_details={"type":"EyeEngine"})
            self._engine = EyeEngine(key_pts_extractor=extractor,
                                     background_model=background_model,
                                     calibrator=calib, key_pts_limiter=key_pts_limiter)
        else: #engine_type == "LightEyeEngine":
            Log.i("Starting Engine", extra_details={"type":"LightEyeEngine"})
            self._engine = LightEyeEngine(calibrator=calib, background_model=background_model,
                                          key_pts_limiter=key_pts_limiter, key_pts_extractor=extractor)

    def get_engine_state(self):
        background_model = self._engine._background_model
        calib = self._engine._calibrator
        return background_model, calib

    def _bytes_to_base64(self, bytes):
        return base64.b64encode(bytes).decode("utf-8")

    def _get_relevant_extractor_engine_type(self, key_points_extractor_string):
        key_points_extractor_string = "hand_silhouette"

        if key_points_extractor_string == 'foot_edge':
            return FootEdgeExtractor(), "EyeEngine"
        if key_points_extractor_string == 'pointing_finger':
            return PointingFingerExtractor(), "EyeEngine"
        if key_points_extractor_string == 'silhouette':
            return SilhouetteExtractor(), "LightEyeEngine"
        if key_points_extractor_string == 'hand_silhouette':
            return HandSilhouetteExtractor(), "LightEyeEngine"
        if key_points_extractor_string == 'random':
            return RandomExtractor(), "EyeEngine"
        return FootEdgeExtractor(), "EyeEngine"

    def _fix_keystones(self, request, type="beam"):
        if type == "beam":
            fix_keystones(self._engine._calibrator)
        elif type == "obie":
            fix_keystones_using_epsons_autokeystones()
        else:
            fix_keystones(self._engine._calibrator)

    def _set_default_keystones(self):
        ProjectorController().change_horizontal_keystone(0)
        ProjectorController().change_vertical_keystone(0)

class ClientServerException(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)
