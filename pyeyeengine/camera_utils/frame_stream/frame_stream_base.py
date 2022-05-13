import typing

from primesense import openni2
from .raw_frame_stream import RawFrameStream
import logging
from pyeyeengine.utilities.global_params import Resolution
from typing import Dict, Generic, TypeVar, Callable

logger = logging.getLogger(__name__)

MappedFrame = TypeVar('MappedFrame')

class FrameStreamBase:
    """
    Wraps a RawFrameStream. Allows to set resolution and returns frames that have been
    mapped from openni2 frames to higher-level frames.
    """
    def __init__(
            self,
            get_device: Callable[[], openni2.Device],
            resolution_name: str,
            mirroring: bool,
            on_mode_changed: typing.Callable[[str], None] = None
    ):
        video_mode = self.calc_video_mode(self.available_resolutions[resolution_name])

        self.raw_frame_stream = RawFrameStream(
            name=self.type_name,
            create_stream=lambda resolution: self.create_video_stream(get_device(), resolution, mirroring),
            video_mode=video_mode,
            on_mode_changed=on_mode_changed,
        )

        self.map_frame_cache = SingleValueCache(self.map_frame)
        self.mirroring = mirroring

    @staticmethod
    def create_video_stream(device: openni2.Device, resolution: Resolution, mirroring: bool) -> openni2.VideoStream:
        raise NotImplementedError()

    @staticmethod
    def map_frame(raw_frame: openni2.VideoFrame, resolution: Resolution) -> MappedFrame:
        raise NotImplementedError()

    @staticmethod
    def calc_video_mode(resolution: Resolution) -> openni2.VideoMode:
        raise NotImplementedError()

    @property
    def available_resolutions(self) -> Dict[str, Resolution]:
        raise NotImplementedError()

    @property
    def type_name(self) -> str:
        return type(self).__name__

    def get_frame(self) -> MappedFrame:
        frame, resolution = self.raw_frame_stream.get_frame_and_resolution()
        return self.map_frame_cache(frame, resolution)

    def wait_for_next_frame(self):
        self.raw_frame_stream.wait_for_next_frame()

    def set_resolution_named(self, resolution_name: str):
        self.set_resolution(self.available_resolutions[resolution_name])

    def set_resolution(self, resolution: Resolution):
        if resolution not in self.available_resolutions.values():
            raise Exception('Unsupported resolution: {}'.format(resolution))

        if self.resolution == resolution:
            return

        logger.debug(
            "Changing {} resolution".format(self.type_name),
            extra={
                "previous": "{}".format(self.resolution),
                "new": "{}".format(resolution)
            }
        )

        self.raw_frame_stream.set_video_mode(self.calc_video_mode(resolution))

        logger.info(
            "{} Resolution Set".format(self.type_name),
            extra={
                "rgb_resolution": self.resolution.string()
            }
        )

    @property
    def stream(self) -> openni2.VideoStream:
        return self.raw_frame_stream.stream

    def start(self):
        self.raw_frame_stream.start()

    def stop(self):
        self.raw_frame_stream.stop()

    @property
    def fps(self):
        return self.raw_frame_stream.fps

    @property
    def resolution(self):
        return self.raw_frame_stream.resolution

    @property
    def mode(self):
        return self.raw_frame_stream.mode

    @property
    def is_running(self):
        return self.raw_frame_stream.is_running

    @property
    def is_stopped(self):
        return self.raw_frame_stream.is_stopped

    @property
    def video_mode(self) -> openni2.VideoMode:
        return self.raw_frame_stream.video_mode

    def __str__(self):
        return '{}({}, mirroring={})'.format(
            self.type_name,
            self.resolution,
            self.mirroring,
        )

    def lazy(self) -> 'FrameStreamBase':
        from .lazy_frame_stream import LazyFrameStream
        return LazyFrameStream(self)


class SingleValueCache:
    """
    Caches the last value computed by a function.
    """
    def __init__(self, func):
        self.func = func
        self.input_and_output = None

    def __call__(self, *args):
        if self.input_and_output is not None and self.input_and_output[0] == args:
            return self.input_and_output[1]
        else:
            output = self.func(*args)
            self.input_and_output = (*args, output)
            return output
