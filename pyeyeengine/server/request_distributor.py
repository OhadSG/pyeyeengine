import base64
import os
import time
import cv2
import json
import traceback
import threading
import tempfile

import numpy as np

from pyeyeengine.utilities import global_params
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
from pyeyeengine.dynamic_surface.dynamic_surface_engine import DynamicSurfaceEngine
from pyeyeengine.projector_controller.projector_controller import ProjectorController
import pyeyeengine.projector_controller.projector_tools as pTools
from pyeyeengine.server.calibration_requests import _calibrate, _get_calibration_images_grid
from pyeyeengine.server.error_logger import ErrorLogger
from pyeyeengine.utilities.helper_functions import *
from pyeyeengine.server.key_points_extractor_socket import KeyPointsExtractorSocket
from pyeyeengine.auto_keystones.auto_keystones_utils import fix_keystones, fix_keystones_using_epsons_autokeystones
import pyeyeengine.server.images_server as images_server
from pyeyeengine.utilities.logging import Log
from pyeyeengine.camera_utils.frame_manager import FrameManager
from pyeyeengine.utilities.file_uploader import FileUploader
from pyeyeengine.utilities import helper_functions as Helper
from pyeyeengine.utilities.preferences import EnginePreferences
from pyeyeengine.utilities.global_params import EngineType
from pyeyeengine.camera_utils.recorder import Recorder
from typing import Dict, Callable, Any
from pyeyeengine.utilities.metrics import Counter
import logging

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_ERROR_LOG = FILE_PATH + "/../error_log.txt"

logger = logging.getLogger(__name__)

request_counter = Counter(
    'request_distributor_request_count',
    namespace='pyeye',
)

class RequestDistributor:
    def __init__(self, screen_resolution=(1280, 800)) -> None:
        super().__init__()
        self.requests = self.create_requests()
        self.is_auto_focusing = False
        self._error_logger = ErrorLogger()
        self.frame_manager = FrameManager()
        self._engine = EyeEngine(frame_manager=self.frame_manager,
                                 key_pts_extractor=FootEdgeExtractor(),
                                 calibrator=AutoCalibrator(frame_manager=self.frame_manager, screen_resolution=screen_resolution),
                                 fps_counter=FPSCounter())
        self._camera_in_use = False
        self.screen_setter = images_server.screen_setter
        self.screen_setter.set_screen_res(screen_width=screen_resolution[0], screen_height=screen_resolution[1])
        self.single_image_calibrator = None
        self.engine_already_ran = False
        self.current_extractor_string = ""
        self._socket = KeyPointsExtractorSocket()
        self.table_mask_not_loaded=True
        self.recorder = None

        threading.Thread(target=self.prepare_projector).start()

    def create_requests(self) -> Dict[str, Callable[[Any], str]]:
        return {
            "ping": self.ping,
            "test_request": self.test_request,
            "record": self.record,
            "get_depth_frame": self.get_depth_frame,
            "get_error_log": self.get_error_log,
            "show_image_in_server": self.show_image_in_server,
            "reset_image_server": self.reset_image_server,
            "reset_frame_manager": self.reset_frame_manager,
            "reload_prefs": self.reload_prefs,
            "upload_debug_files": self.upload_debug_files,
            "reset_server": self.reset_server,
            "get_dynamic_surface": self.get_dynamic_surface,
            "get_dynamic_surface_snapshot": self.get_dynamic_surface_snapshot,
            "get_dynamic_surface_static": self.get_dynamic_surface_static,
            "get_dynamic_surface_static_random": self.get_dynamic_surface_static_random,
            "get_key_points": self.get_key_points,
            "get_template_image": self.get_template_image,
            "get_rbg_as_png": self.get_rbg_as_png,
            "get_grid_image_by_path": self.get_grid_image_by_path,
            "save_rbg_as_png": self.save_rbg_as_png,
            "reload_table_data": self.reload_table_data,
            "get_calibration_images_grid": self.get_calibration_images_grid,
            "calibrate": self.calibrate,
            "calibrate_with_image_server": self.calibrate_with_image_server,
            "calibrate_with_single_image": self.calibrate_with_single_image,
            "auto_focus": self.auto_focus,
            "reset_saved_data": self.reset_saved_data,
            "find_table": self.find_table,
            "get_center_of_mass": self.get_center_of_mass,
            "get_white_background": self.get_white_background,
            "set_screen_res": self.set_screen_res,
            "reset_table_mask": self.reset_table_mask,
            "reset_table_mask_old": self.reset_table_mask_old,
            "get_table_mask": self.get_table_mask,
            "set_table_mask": self.set_table_mask,
            "fix_keystones": self.fix_keystones,
            "fix_keystones_obie": self.fix_keystones_obie,
            "get_monitor_data": self.get_monitor_data,
            "reset_keystones": self.reset_keystones,
            "is_projector_turned_on": self.is_projector_turned_on,
            "turn_on_projector": self.turn_on_projector,
            "turn_off_projector": self.turn_off_projector,
            "get_projector_temp": self.get_projector_temp,
            "get_projector_lamp_hours": self.get_projector_lamp_hours,
            "change_projector_keystones": self.change_projector_keystones,
            "get_fps": self.get_fps,
            "take_high_res_photos": self.take_high_res_photos,
            "get_interactions":self.get_interactions
        }

    def prepare_projector(self):
        # Change driver permissions
        driver_path = FILE_PATH + "/../projector_controller/epscom-driver/"
        os.system("chmod -R 755 {}".format(driver_path))

        projector_controller = ProjectorController()

        if not projector_controller.is_aspect_full():
            projector_controller.set_aspect_to_full()
        if not projector_controller.is_dynamic_color_mode():
            projector_controller.set_dynamic_color_mode()

    def distribute(self, request_json: Dict[str, Any]) -> str:
        try:
            return json.dumps({'status': 'ok', 'data': self._try_serving(request_json)})
        except Exception as e:
            logger.exception(
                'Failed to distribute request',
                extra={
                    "request": {
                        key: value
                        for key, value in request_json.items()
                        if key != 'client'
                    }
                })

            with open(PATH_TO_ERROR_LOG, 'w') as outfile:
                traceback.print_exc(file=outfile)

            self._error_logger.add_error(str(e))
            return json.dumps({'status': 'failed', 'message': str(e)})

    def _try_serving(self, request):
        received_message_text = "Incoming request: '{}'".format(request["name"])
        request_counter.inc({
            'request_name': request['name'],
        })

        if 'params' in request.keys():
            logger.info(received_message_text, extra={"params": request["params"]})
        else:
            logger.info(received_message_text)

        handler = self.requests.get(request['name'])
        if handler is None:
            raise ClientServerException("Command not found: {}".format(request['name']))

        try:
            return handler(request)
        except Exception as e:
            raise ClientServerException('Error in {}'.format(request['name'])) from e

    ####### REQUESTS #######

    def ping(self, request):
        return "Engine is responding"

    def test_request(self, request):
        # FrameManager.getInstance().get_camera_info()
        # FrameManager.getInstance().set_depth_resolution(global_params.Resolution(320, 240))
        # depth_map = FrameManager.getInstance().get_depth_frame()
        return "ok"

    def record(self, request):
        if request["params"]["record"] == 1:
            if self.recorder is not None:
                return 'already recording'

            self.recorder = Recorder(self.frame_manager.rgb_stream)
            self.recorder.start()
        else:
            if self.recorder is None:
                return 'not recording'
            self.recorder.stop()
            self.recorder = None
        return "ok"

    def get_depth_frame(self, _request):
        stream = self.frame_manager.depth_stream
        stream.set_resolution(global_params.DEPTH_HIGH_QUALITY)
        return self._bytes_to_base64(stream.get_frame())

    def get_error_log(self, _request):
        return self._error_logger.get_log()

    def show_image_in_server(self, request):
        if self.screen_setter.image.sum() == 0:
            self.screen_setter.set_image(np.ones_like(self.screen_setter.image) * 123)
        else:
            self.screen_setter.set_image(np.zeros_like(self.screen_setter.image))
        return "ok"

    def reset_image_server(self, _request):
        images_server.reset_thread()
        self.screen_setter.set_screen_res(screen_width=1280, screen_height=800)
        return "reset image server"

    def reset_frame_manager(self, _request):
        self.frame_manager.reset(reason="Client Requested")
        return "ok"

    def reload_prefs(self, request):
        EnginePreferences.getInstance().load()
        logger.info("Engine Preferences", extra=EnginePreferences.getInstance().preferences)
        return "ok"

    def update_offset_x(self,increment):
        EnginePreferences.getInstance().set_text(EnginePreferences.X_OFFSET, "{}".format(str(increment+string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.X_OFFSET)))))
        return "ok"

    def update_offset_y(self,increment):
        EnginePreferences.getInstance().set_text(EnginePreferences.Y_OFFSET, "{}".format(str(increment+string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.Y_OFFSET)))))
        return "ok"

    def update_tilt_x(self,increment):
        EnginePreferences.getInstance().set_text(EnginePreferences.TILT, "{}".format(str(increment+string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.TILT)))))
        return "ok"

    # def update_offset_y(self,increment):
    #     EnginePreferences.getInstance().set_text(EnginePreferences.Y_OFFSET, "{}".format(str(increment+string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.Y_OFFSET)))))
    #     return "ok"

    def update_height(self,increment):
        EnginePreferences.getInstance().set_text(EnginePreferences.DYNAMIC_SURFACE_NEAR_PLANE, "{}".format(str(increment+string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.DYNAMIC_SURFACE_NEAR_PLANE)))))
        EnginePreferences.getInstance().set_text(EnginePreferences.DYNAMIC_SURFACE_FAR_PLANE, "{}".format(str(increment+string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.DYNAMIC_SURFACE_FAR_PLANE)))))
        return "ok"

    def update_scale_y(self,increment):
        EnginePreferences.getInstance().set_text(EnginePreferences.SCALE_Y, "{}".format(str(increment+string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.SCALE_Y)))))
        return "ok"

    def update_scale_x(self,increment):
        EnginePreferences.getInstance().set_text(EnginePreferences.SCALE_X, "{}".format(str(increment+string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.SCALE_X)))))
        return "ok"

    def update_flip(self):
        if EnginePreferences.getInstance().get_switch(EnginePreferences.FLIP)=="false":
            EnginePreferences.getInstance().set_switch(EnginePreferences.FLIP, "true")
        else:
            EnginePreferences.getInstance().set_switch(EnginePreferences.FLIP, "false")
        return "ok"

    def upload_debug_files(self, _request):
        FileUploader.clear_archives_folder()
        calibration_files_folder = FILE_PATH + "/../utilities/local_files/"
        FileUploader.upload_folder(calibration_files_folder,
                                   "calibration_files_{}_{}".format(Log.get_engine_version(), Log.get_system_serial()))

        with tempfile.TemporaryDirectory(prefix='engine-logs') as logs_path:
            Log.prepare_logs_for_upload(logs_path)
            FileUploader.upload_folder(logs_path, "engine_logs_{}_{}".format(Log.get_engine_version(), Log.get_system_serial()))

        return "ok"

    def reset_server(self, request):
        return self._reset_server()

    def get_dynamic_surface(self, request):
        # table_mask = cv2.imread(FILE_PATH + "/../utilities/local_files/table_detection/table_mask.png")
        # mask = self.get_table_mask(request)
        # print(mask)
        # return ""

        if not self._socket.is_active:
            if os.path.exists(FILE_PATH + "/../calibration/table_data_manual.npz"):
                logger.debug("Passing manual mask to DynamicSurface engine")
                table_mask = cv2.imread(FILE_PATH + "/../calibration/table_mask.png")
            else:
                logger.debug("Passing auto mask to DynamicSurface engine")
                table_mask = cv2.imread(FILE_PATH + "/../calibration/table_mask.png")

            self._engine = DynamicSurfaceEngine(
                frame_manager=self.frame_manager,
                calibrator=AutoCalibrator(self.frame_manager, screen_resolution=(1280, 960)),
                table_mask=table_mask
            )
            self._socket.set_engine(self._engine)
            self._socket.add_new_client(request['client'])
            ret_val = self._socket.start()
        else:
            ret_val = self._socket.add_new_client(request['client'])

        self.engine_already_ran = True
        return ret_val

    def get_dynamic_surface_snapshot(self, request):
        if self._engine.engine_type != EngineType.DynamicSurface or self.table_mask_not_loaded:
            if os.path.exists(FILE_PATH + "/../calibration/table_data_manual.npz"):
                logger.debug("Passing manual mask to DynamicSurface engine")
                table_mask = cv2.imread(FILE_PATH + "/../calibration/table_mask.png")
            else:
                logger.debug("Passing auto mask to DynamicSurface engine")
                table_mask = cv2.imread(FILE_PATH + "/../calibration/table_mask.png")

            self._engine = DynamicSurfaceEngine(
                frame_manager=self.frame_manager,
                calibrator=AutoCalibrator(frame_manager=self.frame_manager, screen_resolution=(1280, 800)),
                table_mask=table_mask,
            )
            if table_mask is not None:
                self.table_mask_not_loaded = False

        return self._engine.process_frame()

    def get_dynamic_surface_static(self, request):
        if self._engine.engine_type != EngineType.DynamicSurface:
            if os.path.exists(FILE_PATH + "/../calibration/table_data_manual.npz"):
                Log.d("Passing manual mask to DynamicSurface engine")
                table_mask = cv2.imread(FILE_PATH + "/../calibration/table_mask.png")
            else:
                Log.d("Passing auto mask to DynamicSurface engine")
                table_mask = cv2.imread(FILE_PATH + "/../calibration/table_mask.png")
            self._engine = DynamicSurfaceEngine(calibrator=AutoCalibrator(screen_resolution=(1280, 960)), table_mask=table_mask)
        return self._engine.process_frame_static()

    def get_dynamic_surface_static_random(self, request):
        if self._engine.engine_type != EngineType.DynamicSurface:
            if os.path.exists(FILE_PATH + "/../calibration/table_data_manual.npz"):
                Log.d("Passing manual mask to DynamicSurface engine")
                table_mask = cv2.imread(FILE_PATH + "/../calibration/table_mask.png")
            else:
                Log.d("Passing auto mask to DynamicSurface engine")
                table_mask = cv2.imread(FILE_PATH + "/../calibration/table_mask.png")
            self._engine = DynamicSurfaceEngine(calibrator=AutoCalibrator(screen_resolution=(1280, 960)), table_mask=table_mask)
        return self._engine.process_frame_static_random()

    def get_key_points(self, request):
        if request['params']['key_points_extractor'] == "pointing_finger":
            FileUploader.save_image(self.frame_manager.rgb_stream.get_frame(),
                                    "camera_current_mask.png", "table_detection/")
        return self._get_key_points(request)

    def get_template_image(self, request):
        return self._bytes_to_base64(SingleImageCalibrator.get_template_image())

    def get_rbg_as_png(self, request):
        # self.frame_manager.rgb_stream.set_resolution(global_params.RGB_LOW_QUALITY)
        # self.frame_manager.depth_stream.set_resolution(global_params.DEPTH_MEDIUM_QUALITY)

        depth_stream = self.frame_manager.depth_stream

        depth_map = depth_stream.get_frame()
        depth_map = depth_map - depth_map.min()
        depth_map = depth_map / depth_map.max() * 255
        depth_map_adj = np.uint8(depth_map)
        depth = cv2.applyColorMap(depth_map_adj, cv2.COLORMAP_PARULA)
        rgb = self.frame_manager.rgb_stream.get_frame()

        # resize the depth to be the same as the rgb shape
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

        stack = np.hstack((depth, rgb))
        combined_image = cv2.imencode(".png", stack)[1].tobytes()

        # return self._bytes_to_base64(self.frame_manager.get_rgb_image())
        return self._bytes_to_base64(combined_image)

    def get_grid_image_by_path(self, request):
        return self._bytes_to_base64(self._camera._camera.get_saved_img_as_png(request["params"]["image_path"]))

    def save_rbg_as_png(self, request):
        return self._save_rgb_as_png(request)

    def reload_table_data(self, request):
        return self._engine._calibrator.load_table_data()

    def get_calibration_images_grid(self, request):
        return _get_calibration_images_grid(request)

    def calibrate(self, request):
        _calibrate(self._engine._calibrator)

        return "calibrate done"

    def calibrate_with_image_server(self, request):
        start_time = time.monotonic()
        if self._socket.is_active:
            self._socket.stop()
        screen_resolution = (request["params"]["screen_width"], request["params"]["screen_height"])
        if self._engine.engine_type == EngineType.COM:
            self._engine = EyeEngine(frame_manager=self.frame_manager,
                                     key_pts_extractor=FootEdgeExtractor(),
                                     calibrator=AutoCalibrator(self.frame_manager, screen_resolution=screen_resolution),
                                     fps_counter=FPSCounter())
        if type(self._engine._calibrator) is not AutoCalibrator:
            self._engine._calibrator = AutoCalibrator(self.frame_manager, screen_resolution=screen_resolution)
            if self._engine.engine_type == EngineType.DynamicSurface:
                self._engine._calibrator.load_table_data()
        self._engine._calibrator.calibrate(mode=request["params"]["mode"],
                                           screen_setter=self.screen_setter)
        print("calibrate: %f (s)" % (time.monotonic() - start_time))
        self.frame_manager.rgb_stream.set_resolution_named('small')
        self.table_mask_not_loaded = True

        return "calibrated sucessfully" if self._engine._calibrator.calibrate_success else "calibrated failed"

    def calibrate_with_single_image(self, request):
        start_time = time.monotonic()

        if self._socket.is_active:
            self._socket.stop()

        screen_resolution = global_params.Resolution(request["params"]["screen_width"],
                                                     request["params"]["screen_height"])

        if self.single_image_calibrator is None:
            self.single_image_calibrator = SingleImageCalibrator(
                frame_manager=self.frame_manager,
                screen_resolution=global_params.SCREEN_RESOLUTION
            )

        self.single_image_calibrator.screen_resolution = screen_resolution

        self._engine._calibrator = self.single_image_calibrator
        self._engine._calibrator.calibrate(mode=request["params"]["mode"])

        Log.d("Calibration duration", extra_details={"duration": "{:.1f}".format(time.monotonic() - start_time)})
        self.frame_manager.rgb_stream.set_resolution(global_params.RGB_LOW_QUALITY)
        return "calibrated sucessfully" if self._engine._calibrator.calibrate_success else "calibrated failed"

    def auto_focus(self, request):
        if pTools.supports_function(pTools.FUNCTION_FOCUS):
            if self.is_auto_focusing == False:
                Log.i("Auto-focus requested", flow="auto_focus")

                self.is_auto_focusing = True
                success = pTools.auto_focus()
                self.is_auto_focusing = False

                Log.i("Auto-focus result", extra_details={"success": "{}".format(success)}, flow="auto_focus")

                if success:
                    return "ok"
                else:
                    return "fail"
            else:
                return "fail"
        else:
            return "fail"

    def reset_saved_data(self, request):
        return self.reset_saved_table_data(request)

    def find_table(self, request):
        return self.find_table()

    def get_center_of_mass(self, request):
        return self._get_center_of_mass(request)

    def get_white_background(self, request):
        return self._bytes_to_base64(cv2.imencode(".png",
                                                  self._engine._calibrator.generate_blurry_white_screen()
                                                  )[1].tobytes())

    def set_screen_res(self, request):
        self._engine._calibrator.screen_width = request["params"]["screen_width"]
        self._engine._calibrator.screen_height = request["params"]["screen_height"]
        return "success"

    def reset_table_mask(self, request):
        Log.i("Resetting Table Mask")
        # return "success"
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, "../calibration")

        mask_files = ["/playing_mask.png",
                      "/table_data.npz",
                      "/table_data_manual.npz",
                      "/table_detected.png",
                      "/table_detected_rough.png",
                      "/table_mask.png"]

        for file in mask_files:
            if os.path.exists(path + file):
                os.remove(path + file)

        return "success"

    def reset_table_mask_old(self, request):
        Log.i("Resetting Table Mask")
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, "../calibration")
        if os.path.exists(path + "/table_data_manual.npz"):
            os.remove(path + "/table_data_manual.npz")

        return "success"

    def get_table_mask(self, request):
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
                "shape": shape,
            }

        table_mask, rect_display = export_data_for_android(self._engine._calibrator)
        table_mask_with_alpha = np.concatenate([np.tile(table_mask[:, :, np.newaxis], (1, 1, 3)),
                                                255 - table_mask[:, :, np.newaxis]], axis=2)

        table_data = {
            "image": self._bytes_to_base64(cv2.imencode(".png", table_mask_with_alpha)[1].tobytes()),
            "top": "%d" % (rect_display[1]),
            "left": "%d" % (rect_display[0]),
            "bottom": "%d" % (rect_display[3]),
            "right": "%d" % (rect_display[2]),
            "shape": "" if self._engine._calibrator.table_shape is None else self._engine._calibrator.table_shape,
        }

        top_left = Helper.Point(rect_display[0], rect_display[1])
        bottom_right = Helper.Point(rect_display[2], rect_display[3])
        area = Helper.calculate_rectangle_area(top_left, bottom_right)
        screen_area = self._engine._calibrator.screen_width * self._engine._calibrator.screen_height
        percent_of_mask = (area / screen_area) * 100

        # print("Mask Points: {} {}".format(top_left.string(), bottom_right.string()))
        # a, b, c, d = Helper.get_mask_corners(top_left, bottom_right)
        # print("Points: {} {} {} {}".format(a.string(), b.string(), c.string(), d.string()))
        # is_valid_rectangle = Helper.is_rectangle_any_order(a, b, c, d)

        Log.d("Table mask covers {:.1f}% of screen space".format(percent_of_mask))

        # print("Is Valid: {}".format(is_valid_rectangle))

        if percent_of_mask < 10.0:
            raise Exception("Table not found")

        FileUploader.save_text_file("{}".format(table_data), "table_data.txt", folder="table_detection/")

        return table_data

    def set_table_mask(self, request):

        if request["params"]["shape"] == 'circle':
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

            Log.i("Setting Manual Table Mask", extra_details={"shape": "circle",
                                                              "center": "{}".format(center),
                                                              "radius": "{}".format(radius)})
            np.savez(path + "/table_data_manual", table_shape='circle',
                     table_mask=table_mask, edge_params={"top": center[0] - radius,
                                                         "bottom": center[0] + radius,
                                                         "left": center[1] - radius,
                                                         "right": center[1] + radius})

        elif request["params"]["shape"] == 'rectangle':
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

            if self.single_image_calibrator is None:
                self.single_image_calibrator = SingleImageCalibrator(screen_resolution=global_params.SCREEN_RESOLUTION)

            mask_width = bottom_left[1] - top_left[1]
            mask_height = top_right[0] - top_left[0]

            with open(path + "/table_mask_size.txt", "w") as file:
                file.write(json.dumps({"width": mask_width, "height": mask_height}))
                file.close()

            print("Mask Size: {}".format((mask_width, mask_height)))

            Log.i("Setting Manual Table Mask", extra_details={"shape": "rectangle",
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

    def fix_keystones(self, request):
        self._fix_keystones(request, "beam")
        return ""

    def fix_keystones_obie(self, request):
        self._fix_keystones(request, "obie")
        return ""

    def get_monitor_data(self, request):
        return self._get_monitor_data()

    def reset_keystones(self, request):
        self._set_default_keystones()
        return ""

    def is_projector_turned_on(self, request):
        return ProjectorController().is_turned_on()

    def turn_on_projector(self, request):
        return ProjectorController().turn_on()

    def turn_off_projector(self, request):
        return ProjectorController().turn_off()

    def get_projector_temp(self, request):
        return ProjectorController().get_temp()

    def get_projector_lamp_hours(self, request):
        return ProjectorController().get_lamp_hours()

    def change_projector_keystones(self, request):
        return ProjectorController().change_vertical_keystone(request["params"]["degrees"])

    def get_fps(self, request):
        return "%f" % self._engine._fps_counter.fps

    def take_high_res_photos(self, request):
        for _ in range(5):
            self.frame_manager.rgb_stream.get_frame_as_png()
        return ""

    ####### REQUESTS #######

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
                "camera_serial_number": self.frame_manager.get_serial_number() }

    def reset_saved_table_data(self, request):
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
        self.frame_manager.rgb_stream.save_frame_as_png(request["params"]["save_path"])
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
        self.frame_manager.reset(reason="Server Reset")
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

    def get_interactions(self, request):
        interactions = self._socket.get_interactions()
        print(interactions)
        return json.dumps({"interactions":interactions})

    def init_center_of_mass(self):
        self._engine = COMEngine()
        self._engine.define_bg()
        Log.i("Center of mass engine initialized")

    def _get_key_points(self, request):
        if self.single_image_calibrator is None:
            self.single_image_calibrator = SingleImageCalibrator(
                frame_manager=self.frame_manager,
                screen_resolution=global_params.SCREEN_RESOLUTION
            )
        # TODO: IF ENGINE IS COM, THINGS NEED TO BE HANDLED DIFFERENTLY
        logger.debug(
            "Engine Status",
            extra={
                "engine_ran": "{}".format(self.engine_already_ran),
                "engine_fps": "{}".format(int(self._engine._fps_counter.fps)),
                "frame_manager_fps": "{}".format(self.frame_manager.depth_stream.fps)
            }
        )

        # if (self.engine_already_ran and self._engine._fps_counter.fps < 10):
        #     Log.w("Engine FPS Dropped",
        #           extra_details={"engine_fps": "{}".format(self._engine._fps_counter.fps)})
        #     self._socket.stop()

        if self.frame_manager.depth_stream.fps < 3:
            logger.warning(
                "Camera FPS Dropped",
                extra={"frame_manager_fps": "{}".format(self.frame_manager.depth_stream.fps)}
            )

        extractor_string = request['params']['key_points_extractor']

        if not self._socket.is_active or self.current_extractor_string is not extractor_string:
            Log.d("Using a new engine", extra_details={"current_extractor": self.current_extractor_string, "new_extractor": extractor_string})
            self.current_extractor_string = extractor_string
            self.init_engine_for_get_key_points(request)
            self._socket.set_engine(self._engine)
            self._socket.add_new_client(request['client'])
            ret_val = self._socket.start()
        else:
            Log.d("Using an existing engine", extra_details={"current_extractor": self.current_extractor_string, "new_extractor": extractor_string})
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
            self._engine = EyeEngine(frame_manager=self.frame_manager,
                                     key_pts_extractor=extractor,
                                     background_model=background_model,
                                     calibrator=calib, key_pts_limiter=key_pts_limiter)
        else: #engine_type == "LightEyeEngine":
            Log.i("Starting Engine", extra_details={"type":"LightEyeEngine"})
            self._engine = LightEyeEngine(frame_manager=self.frame_manager,
                                          calibrator=calib,
                                          background_model=background_model,
                                          key_pts_limiter=key_pts_limiter,
                                          key_pts_extractor=extractor)

    def get_engine_state(self):
        background_model = self._engine._background_model
        calib = self._engine._calibrator
        return background_model, calib

    def _bytes_to_base64(self, bytes):
        return base64.b64encode(bytes).decode("utf-8")

    def _get_relevant_extractor_engine_type(self, key_points_extractor_string):
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
