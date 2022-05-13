import numpy as np
import cv2
from primesense import openni2  # , nite2
from primesense import _openni2 as c_api
from primesense.openni2 import IMAGE_REGISTRATION_DEPTH_TO_COLOR
import platform
import inspect
import os
import ctypes

from pyeyeengine.camera_utils.camera_reader import CameraReader

if __name__ == '__main__':
    camera = CameraReader()
    answer = camera.device.get_property(c_api.ONI_DEVICE_PROPERTY_SERIAL_NUMBER, (ctypes.c_char * 100))
    print(answer.value)
