import threading
import logging
from typing import Callable
import sys

from primesense import openni2
from pyeyeengine.utilities.global_params import Resolution
from pyeyeengine.utilities.threading_tools import ExecTimeChecker
from pyeyeengine.utilities.metrics import Counter
from pyeyeengine.utilities.timeout import Timeout

from ..fps_monitor import FpsMonitor


frame_counter = Counter(
    name='frame_received_count',
    namespace='pyeye',
)
logger = logging.getLogger(__name__)

class RawFrameStream:
    """
    Manages an openni2.VideoStream.
    """
    def __init__(
            self,
            name: str,
            create_stream: Callable[[Resolution], openni2.VideoStream],
            video_mode: openni2.VideoMode,
            on_mode_changed: Callable[[str], None] = None
    ):
        self.name = name
        self.create_stream = create_stream
        self.latest_frame = None
        self.on_mode_changed = on_mode_changed
        self.video_mode = video_mode

        self.received_frame_event = threading.Event()
        self.lock = threading.RLock()
        self.sanity_thread = None

        self.stream = None
        self.mode = 'stopped'
        self.fps_monitor = FpsMonitor()

    def get_frame(self) -> openni2.VideoFrame:
        return self.get_frame_and_resolution().frame

    def wait_for_next_frame(self):
        self.received_frame_event.clear()
        self.received_frame_event.wait()

    def get_frame_and_resolution(self) -> 'FrameAndResolution':
        if self.mode != 'running':
            self.start()

        res = self.latest_frame
        if res is None:
            self.received_frame_event.wait()
            res = self.latest_frame

        return res

    @property
    def resolution(self):
        return get_video_mode_resolution(self.video_mode)

    def set_video_mode(self, video_mode: openni2.VideoMode):
        with self.lock:
            try:
                if self.mode == 'stopped':
                    self.video_mode = video_mode
                    if self.stream is not None:
                        self.stream.set_video_mode(video_mode)
                else:
                    self._set_mode('setting-mode')
                    self.stream.stop()
                    self.latest_frame = None
                    self.received_frame_event.clear()
                    self.stream.set_video_mode(video_mode)
                    self.video_mode = video_mode
                    self.stream.start()
                    self._set_mode('running')
            except:
                logger.exception('Failed to set mode')
                self._set_mode('error')
                raise

    @property
    def is_running(self):
        return self.mode == 'running'

    @property
    def is_stopped(self):
        return self.mode == 'stopped'

    def __on_frame_received(self, stream: openni2.VideoStream):
        frame = stream.read_frame()
        resolution = get_video_mode_resolution(stream.video_mode)

        self.fps_monitor.on_frame_received()
        frame_counter.inc({
            'name': self.name,
            'resolution': str(self.resolution)
        })

        if self.sanity_thread:
            self.sanity_thread.tick()

        mode = self.mode
        if mode != 'running':
            logger.warning('Ignoring frame because the stream is in {} mode'.format(mode))
            return

        with self.lock:
            self.latest_frame = FrameAndResolution(frame, resolution)

        self.received_frame_event.set()

    def start(self):
        if self.mode == 'running':
            return

        with self.lock:
            if self.mode == 'running':
                return

            if self.mode != 'stopped':
                raise Exception('Must be stopped')

            self._set_mode('starting')
            try:
                if self.sanity_thread is None or not self.sanity_thread.isAlive():
                    logger.info("Starting watch dog")
                    self.sanity_thread = ExecTimeChecker(name="FrameManagerWatchdog",
                                                         caller="__on_frame_received",
                                                         message="Didn't receive a frame in a while",
                                                         callback=self.reset,
                                                         time_to_wait=30)
                    self.sanity_thread.start()
                if self.stream is None:
                    self.stream = self.create_stream(self.resolution)
                    self.stream.register_new_frame_listener(self.__on_frame_received)
                    self.stream.start()
                else:
                    self.stream.start()
                self._set_mode('running')
            except:
                logger.exception('Failed to start stream')
                self._set_mode('error')
                raise

    def stop(self):
        with Timeout(60, self.__on_stop_timeout):
            with self.lock:
                if self.mode == 'stopped':
                    return

                self._set_mode('stopping')
                try:
                    self.stream.stop()
                    logger.info('Stopped raw stream')

                    # Sometimes this never completes and I've no idea why,
                    # so we commented it out
                    # self.stream.close()
                    # logger.info('Closed raw stream')
                    # self.stream = None

                    self.sanity_thread = None
                    self._set_mode('stopped')
                except:
                    logger.exception('Failed to stop stream')
                    self._set_mode('error')
                    raise
                return "done"

    def reset(self):
        logger.info("Resetting raw frame stream")
        self.stop()
        self.start()

    @property
    def fps(self):
        return self.fps_monitor.fps

    def __on_stop_timeout(self, timeout):
        logger.error('{} timed out trying to stop the stream after {} seconds'.format(self.name, timeout))

    def _set_mode(self, mode):
        if self.mode == mode:
            return

        logger.info('{} changing from {} to {}'.format(self, self.mode, mode))
        self.mode = mode

        if self.on_mode_changed is not None:
            self.on_mode_changed(mode)

    def __str__(self):
        return self.name

class FrameAndResolution:
    def __init__(self, frame: openni2.VideoFrame, resolution: Resolution):
        self.frame = frame
        self.resolution = resolution

    def __eq__(self, other):
        return isinstance(other, FrameAndResolution) and self.as_tuple() == other.as_tuple()

    def as_tuple(self):
        return (self.frame, self.resolution)

    def __iter__(self):
        return iter(self.as_tuple())

def get_video_mode_resolution(video_mode: openni2.VideoMode) -> Resolution:
    return Resolution(video_mode.resolutionX, video_mode.resolutionY)