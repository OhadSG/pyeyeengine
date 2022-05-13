import cv2
import numpy as np

from pyeyeengine.auto_keystones.auto_keystones_utils import find_rectification_tform, find_trapazoid_size_angles
from pyeyeengine.calibration.auto_calibrator import AutoCalibrator
from pyeyeengine.camera_utils.camera_reader import CameraReader


if __name__ == '__main__':
    calibrator = AutoCalibrator()
    calibrator.calibrate(recollect_imgs=True)

    cam = CameraReader(display=False, resxy=(640, 480))
    keep_running = True
    dmap = cam.get_depth()
    rgb = cam.get_rgb()
    tform_rectify = find_rectification_tform(dmap)

    bottom_left, bottom_right, top_left, top_right = calibrator.get_display_corners_on_cam()

    bottom_left, bottom_right = cv2.transform(bottom_left, tform_rectify),  cv2.transform(bottom_right, tform_rectify)
    top_left, top_right = cv2.transform(top_left, tform_rectify), cv2.transform(top_right, tform_rectify)

    theta = find_trapazoid_size_angles(bottom_left, bottom_right, top_left, top_right)









