from primesense import _openni2 as c_api
from primesense import openni2
import numpy as np
import sys
import logging
from pyeyeengine.camera_utils.error import CameraFailed
from .frame_stream_base import FrameStreamBase
from pyeyeengine.utilities.global_params import Resolution
import numpy
import typing

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 8):
    from typing import NewType
    DepthFrame = NewType('DepthFrame', numpy.array)
else:
    DepthFrame = typing.Any

class DepthFrameStream(FrameStreamBase):
    available_resolutions = {
        'small': Resolution(160, 120),
        'medium': Resolution(320, 240),
        'large': Resolution(640, 480),
    }

    @staticmethod
    def create_video_stream(device: openni2.Device, resolution: Resolution, mirroring: bool):
        depth_stream = device.create_depth_stream()

        try:
            depth_stream.set_video_mode(DepthFrameStream.calc_video_mode(resolution))
        except Exception as e:
            logger.exception("Exception caught trying to change depth resolution to {}".format(resolution))
            raise CameraFailed("Failed to change camera resolution to depth: {}".format(resolution)) from e
        depth_stream.set_mirroring_enabled(mirroring)

        return depth_stream

    @staticmethod
    def map_frame(raw_frame: openni2.VideoFrame, resolution: Resolution) -> DepthFrame:
        buffer = raw_frame.get_buffer_as_uint16()
        depth_array = np.fromstring(buffer, dtype=np.uint16)
        return depth_array.reshape(resolution.height, resolution.width)

    @staticmethod
    def calc_video_mode(resolution: Resolution) -> openni2.VideoMode:
        return c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                  resolutionX=resolution.width,
                                  resolutionY=resolution.height,
                                  fps=30)
