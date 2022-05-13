import os
import platform
import threading
import ctypes
import re
import subprocess
from primesense import openni2
from primesense import _openni2 as c_api
import logging

from pyeyeengine.utilities.read_write_lock import RWLock
from pyeyeengine.utilities.logging import Log
from pyeyeengine.utilities.init_openni import init_openni
from pyeyeengine.camera_utils.frame_stream.depth_frame_stream import DepthFrameStream
from pyeyeengine.camera_utils.frame_stream.rgb_frame_stream import RgbFrameStream
from .error import CameraFailed

MAX_RESOLUTION_SET_RETRIES = 10
CAMERA_FPS = 30.0
RESTART_DELAY = 4
MIN_CAMERA_SERIAL = 17122000000
MAX_CAMERA_SERIAL = 18021000000

logger = logging.getLogger(__name__)

class FrameManager:
    __instance = None

    @staticmethod
    def getInstance():
        instance = FrameManager.__instance
        if instance is None:
            instance = FrameManager()
            FrameManager.__instance = instance
        return instance

    def __init__(self, register=True, mirroring=False):
        if FrameManager.__instance is not None:
            raise Exception('Only one frame manager can exist at once.')

        logger.info(
            "Initializing FrameManager (Thread: {} ID: {})".format(
                threading.current_thread().name,
                threading.current_thread().ident
            )
        )

        self.initialized = False
        self.ready = False
        self.resetting = False
        self.camera_found = False
        self.lock = RWLock()
        self.type = "rgbd"
        self.record = False
        self.register = register
        self.sanity_thread_rgb = None
        self.sanity_thread_depth = None
        self.is_resetting = True
        self.retry_attempts = 0
        self.initialized = True
        self.device = None

        self.depth_stream = DepthFrameStream(
            self.get_device,
            'medium',
            mirroring,
            on_mode_changed=self._on_stream_mode_changed
        ).lazy()

        self.rgb_stream = RgbFrameStream(
            self.get_device,
            'medium',
            mirroring,
            on_mode_changed=self._on_stream_mode_changed,
        ).lazy()

        FrameManager.__instance = self

    def get_device(self):
        if self.device is None:
            self.device = FrameManager.__init_device()
        return self.device

    def _on_stream_mode_changed(self, _mode):
        self.close_device_if_not_used()

    def close_device_if_not_used(self):
        if self.device is not None and self.depth_stream.is_stopped and self.rgb_stream.is_stopped:
            logger.info('Closing device because all streams are stopped')
            self.device.close()
            self.device = None

    @staticmethod
    def unload_driver():
        Log.d("Unloading OpenNI driver")
        openni2.unload()

    @staticmethod
    def __init_device():
        try:
            init_openni()
            device = openni2.Device.open_any()
        except Exception as e:
            raise CameraFailed("Failed to init openni device") from e

        return device

    def reset(self, reason: str):
        logger.info('Resetting because {}'.format(reason))
        self.rgb_stream.stop()
        self.depth_stream.stop()

    def check_if_camera_valid(self):
        serial_number = self.get_serial_number()

        if "17122" in serial_number:
            Log.e("Camera Serial Invalid")
            raise CameraFailed("Camera Serial Invalid (SN 17122xxxxxx)")

        if platform.system().lower() == "darwin":
            pass
        else:
            if re.search('arm', platform.machine(), re.IGNORECASE):
                try:
                    answer = subprocess.check_output('sudo lsusb -d 2bc5:0401', shell=True, timeout=10)
                except:
                    try:
                        answer = subprocess.check_output('sudo lsusb -d 1d27:0609', shell=True, timeout=10)
                    except:
                        answer = ""
                if len(answer) == 0:
                    Log.e("Camera Not Found")
                    raise CameraFailed("Camera Not Found")
            else:
                FILE_PATH = os.path.dirname(os.path.realpath(__file__))
                cmd = "%s\SimpleReadFW\Release\SimpleReadFW.exe" % (FILE_PATH)
                timeout = 1
                with subprocess.Popen(cmd, stdout=-1) as process:
                    try:
                        answer, stderr = process.communicate(None, timeout=timeout)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        answer, stderr = process.communicate()
                        answer = str(answer)

                if "UNKNOW TYPE" in answer:
                    Log.e("Unkown Camera Type")
                    # raise CameraFailed("No orbbec device found ( there is probably a primesense camera attached ).")
        return serial_number

    def get_serial_number(self):
        camera_serial_raw = self.get_device().get_property(c_api.ONI_DEVICE_PROPERTY_SERIAL_NUMBER, (ctypes.c_char * 100)).value
        camera_serial_string = str(camera_serial_raw).split("'")[1]
        return camera_serial_string

######################## Notes ########################

# 1. Regarding Depth Image Fetching
"""
Returns numpy ndarrays representing the raw and ranged depth images.
Outputs:
    dmap:= distancemap in mm, 1L ndarray, dtype=uint16, min=0, max=2**12-1 ( in mm from cam )
Note1:
    fromstring is faster than asarray or frombuffer
Note2:
    .reshape(120,160) #smaller image for faster response
            OMAP/ARM default video configuration
    .reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
            Requires .set_video_mode
"""
