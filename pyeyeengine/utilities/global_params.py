from enum import Enum

class Resolution():
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def tuple(self):
        return (self.width, self.height)

    def string(self):
        return "{}x{}".format(self.width, self.height)

    def is_equal(self, to_resolution):
        return (self.width == to_resolution.width and self.height == to_resolution.height)

DEFAULT_CAMERA_RESOLUTION = Resolution(320, 240)
SCREEN_RESOLUTION = Resolution(1280, 960)
PROJECTOR_RESOLUTION = Resolution(1280, 800)
DEFAULT_DEPTH_IMAGE_SIZE = Resolution(640, 480)
OLD_CAMERA_RGB_IMAGE_SIZE = Resolution(1280, 1024)
CORRECTED_RGB_IMAGE_SIZE = Resolution(1280, 1024)

# RGB Resolutions
RGB_LOW_QUALITY = Resolution(320, 240)
RGB_MEDIUM_QUALITY = Resolution(640, 480)
RGB_HIGH_QUALITY = Resolution(1280, 1024)

# Depth Resolutions
DEPTH_LOW_QUALITY = Resolution(160, 120)
DEPTH_MEDIUM_QUALITY = Resolution(320, 240)
DEPTH_HIGH_QUALITY = Resolution(640, 480)

# class RGBResolution(Enum):
#     low = Resolution(320, 240)
#     medium = Resolution(640, 480)
#     high = Resolution(1280, 1024)
#
# class DepthResolution(Enum):
#     low = Resolution(160, 120)
#     medium = Resolution(320, 240)
#     high = Resolution(640, 280)

# We minimize the calibration result by 4, because the games work with a
# smaller scale rather than the full projection resolution
CALIBRATION_SCALE_FACTOR = 4

# Standard resolution for the AstraOrbbec class is defined as 640 x 480
CAMERA_SCALE_FACTOR = 640 / DEFAULT_DEPTH_IMAGE_SIZE.width

CAMERA_ANGLE_OBIE = 50 #The offset angle for the camera in Obie devices.

class EngineType(Enum):
     COM = 1
     LightTouch = 2
     EyeEngine = 3