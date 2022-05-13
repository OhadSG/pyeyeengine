import cv2
import time

import numpy as np

from pyeyeengine.calibration.auto_calibrator import AutoCalibrator
from pyeyeengine.calibration.find_max_playing_mask import find_max_playing_mask
from pyeyeengine.camera_utils.rgb_camera_reader import RGBCameraReader

if __name__ == '__main__':

    calibrator = AutoCalibrator()
    # calibrator.get_table_mask()
    # calibrator.calibrate(recollect_imgs=True )


    camera = RGBCameraReader(display=False, resxy=(640, 480))
    keep_running = True

    cv2.namedWindow("mask_for_display", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("mask_for_display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    to_display = np.uint8(np.ones((calibrator.screen_height, calibrator.screen_width, 3)) * 255)
    # to_display = find_max_playing_mask(calibrator, is_display=True)
    while keep_running:
        start = time.time()
        key = cv2.waitKey(1)
        if key == 27:
            keep_running = False

        rgb = camera.get_rgb()
        ret, corners = cv2.findChessboardCorners(rgb, (4, 4), None)
        to_display_copy = to_display.copy()

        if ret:
            transformed_corners = calibrator.transfrom_points_cam_to_display(np.squeeze(corners, axis=1) / 2)
            for pt_num in range(transformed_corners.shape[0]):
                to_display_copy = cv2.circle(to_display_copy,
                                             (transformed_corners[pt_num, 0], transformed_corners[pt_num, 1]), 1,
                                             (0, 0, 255), -1)

        cv2.imshow("mask_for_display", to_display_copy)
        cv2.imshow("rgb", rgb)
        cv2.waitKey(1)
        end = time.time()
        print(end - start)

    cv2.destroyAllWindows()
    camera.stop()
