import numpy
import sys
from primesense import _openni2 as c_api
from primesense import openni2
import numpy as np
from pyeyeengine.camera_utils.error import CameraFailed
import cv2
from .frame_stream_base import FrameStreamBase
from pyeyeengine.utilities.global_params import Resolution
import logging
import typing

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 8):
    from typing import NewType
    RgbFrame = NewType('RgbFrame', numpy.array)
else:
    RgbFrame = typing.Any

class RgbFrameStream(FrameStreamBase):
    available_resolutions = {
        'small': Resolution(320, 240),
        'medium': Resolution(640, 480),
        'large': Resolution(1280, 1024),
    }

    def get_frame_as_png(self):
        return cv2.imencode(".png", self.get_frame())[1].tobytes()

    def save_frame_as_png(self, image_path):
        rgb = self.get_frame_as_png()
        return cv2.imwrite(image_path, rgb)

    @staticmethod
    def create_video_stream(device: openni2.Device, resolution: Resolution, mirroring: bool):
        rgb_stream = device.create_color_stream()

        try:
            rgb_stream.set_video_mode(RgbFrameStream.calc_video_mode(resolution))
        except Exception as e:
            logger.exception("Exception caught trying to change camera resolution to {}".format(resolution))
            raise CameraFailed("Failed to change RGB camera resolution to {}".format(resolution)) from e
        rgb_stream.set_mirroring_enabled(mirroring)

        return rgb_stream

    @staticmethod
    def map_frame(frame: openni2.VideoFrame, resolution: Resolution) -> RgbFrame:
        buffer = frame.get_buffer_as_uint8()
        bgr_array = np.fromstring(buffer, dtype=np.uint8)
        bgr = bgr_array.reshape(resolution.height, resolution.width, 3)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    @staticmethod
    def calc_video_mode(resolution):
        return c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                  resolutionX=resolution.width,
                                  resolutionY=resolution.height,
                                  fps=30)

