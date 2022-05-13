import numpy as np
import cv2
from primesense import openni2  # , nite2
from primesense import _openni2 as c_api
from primesense.openni2 import IMAGE_REGISTRATION_DEPTH_TO_COLOR
import platform
import inspect
import os
import pyeyeengine.utilities.global_params as Globals

current_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
distribution = current_dir + "/../../../OpenNI2/Redist"

class LocalFrameManager():
    def __init__(self, rgb_resolution=Globals.RGB_MEDIUM_QUALITY, depth_resolution=Globals.DEPTH_MEDIUM_QUALITY):
        self.rgb_resolution = rgb_resolution
        self.depth_resolution = depth_resolution
        self.device = None
        self.depth_stream = None
        self.rgb_stream = None
        self.prepare_streams()

    def prepare_streams(self):
        distribution = os.path.dirname(os.path.realpath(__file__)) + "/../../../OpenNI2/Redist"

        if "arm" in platform.machine():
            distribution += "_ARM"
        elif "Linux" in platform.system():
            distribution += "_Linux_64"
        elif "Darwin" in platform.system():
            distribution += "_Mac_64"
        else:
            if platform.architecture()[0] == "32bit":
                distribution += "_WIN_32"
            else:
                distribution += "_WIN_64"

        openni2.initialize(distribution)

        if (openni2.is_initialized()):
            print("openNI2 initialized")

        ## Register the device
        self.device = openni2.Device.open_any()

        ## Create the streams
        self.depth_stream = self.device.create_depth_stream()
        self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM,
                                                       resolutionX=self.depth_resolution.width,
                                                       resolutionY=self.depth_resolution.height,
                                                       fps=30))
        self.depth_stream.start()

        self.rgb_stream = self.device.create_color_stream()
        self.rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                     resolutionX=self.rgb_resolution.width,
                                                     resolutionY=self.rgb_resolution.height,
                                                     fps=30))
        self.rgb_stream.start()

        self.device.set_image_registration_mode(IMAGE_REGISTRATION_DEPTH_TO_COLOR)


    def get_depth_frame(self):
        depth_frame = self.depth_stream.read_frame()
        buffer = depth_frame.get_buffer_as_uint16()
        depth_array = np.fromstring(buffer, dtype=np.uint16)
        depth_map = depth_array.reshape(self.depth_resolution.height, self.depth_resolution.width)
        max_depth_map = np.max(depth_map)
        min_depth_map = np.min(depth_map)
        depth_map_adj = np.uint8(((depth_map - min_depth_map) / (np.max((max_depth_map - min_depth_map, 1)))) * 255)
        depth = cv2.applyColorMap(depth_map_adj, cv2.COLORMAP_TWILIGHT_SHIFTED)
        # depth = cv2.resize(depth, (640, 480))
        return depth

    def get_rgb_frame(self):
        bgr = np.fromstring(self.rgb_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint8).reshape(
            self.rgb_resolution.height, self.rgb_resolution.width, 3)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # rgb = cv2.resize(rgb, (640, 480))
        return rgb