import numpy as np
import cv2
from primesense import openni2  # , nite2
from primesense import _openni2 as c_api
import platform
import inspect
import os



current_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
distribution = current_dir + "/../OpenNI2/Redist"
if "arm" in platform.machine():
    distribution += "_ARM"
else:
    if platform.architecture()[0] == "32bit":
        distribution += "_WIN_32"
    else:
        distribution += "_WIN_64"


openni2.initialize(distribution)
if (openni2.is_initialized()):
    print("openNI2 initialized")

## Register the device
device = openni2.Device.open_any()

## Create the streams
rgb_stream = device.create_color_stream()

rgb_stream.set_video_mode(
            c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                               resolutionX=1280,
                               resolutionY=1024, fps=30))
rgb_stream.start()

while True:
    bgr = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint8).reshape(
        1024, 1280, 3)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cv2.imshow("rgb", rgb)
    cv2.waitKey(1)

