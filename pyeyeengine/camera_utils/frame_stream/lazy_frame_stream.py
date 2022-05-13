import os
import threading
import time
from pyeyeengine.utilities.read_write_lock import RWLock
from pyeyeengine.utilities.logging import Log
import logging

logger = logging.getLogger(__name__)

DEFAULT_MAX_IDLE_TIME = float(os.getenv('PYEYE_LAZY_FRAME_STREAM_MAX_IDLE_TIME', default='60'))

class LazyFrameStream:
    """
    Wraps a stream and closes it after it hasn't been used for some time.
    """
    def __init__(
            self,
            stream,
            max_idle_time = DEFAULT_MAX_IDLE_TIME,
            check_interval = 1,
    ):
        self.stream = stream
        self.last_used_time = None
        self.max_idle_time = max_idle_time
        self.check_interval = check_interval
        self.thread = threading.Thread(target=self.__thread_main, name='LazyFrameStreamThread')
        self.thread.start()

    def __getattr__(self, item):
        self.last_used_time = time.monotonic()
        return getattr(self.stream, item)

    def __thread_main(self):
        while True:
            time.sleep(self.check_interval)

            try:
                if self.stream.mode != 'running' or self.last_used_time is None:
                    continue

                time_since_last_get_frame = time.monotonic() - self.last_used_time
                if time_since_last_get_frame > self.max_idle_time:
                    logger.info("Closing stream {} because it was inactive for {}s".format(self.stream, time_since_last_get_frame))
                    self.stream.stop()
            except:
                logger.exception('Error in LazyFrameStream thread')
