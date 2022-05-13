import cv2
import numpy as np

from pyeyeengine.camera_utils.camera_reader import CameraReader
from pyeyeengine.object_detection.table_detector import TableDetector

if __name__ == '__main__':
    table_detector = TableDetector()
    cam = CameraReader()

    cv2.namedWindow("white_background", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("white_background", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    white_screen = np.float32(np.zeros((240, 320, 3)))
    white_screen[5:-5, 5:-5, :] = 1
    white_screen = np.uint8(cv2.filter2D(white_screen, -1, np.ones((21, 21)) / 21 / 21 * 255))
    cv2.imshow("white_background", white_screen)
    cv2.waitKey(1)

    while True:
        depth_map = cv2.resize(cam.get_depth(), (1280, 960))
        rgb = cv2.resize(cam.get_rgb(), (1280, 960))
        contours = table_detector.find_potential_table_contours(depth_map)
        table_contour = table_detector.decide_which_contour_is_the_table(contours, depth_map)
        if table_contour is not None:
            rgb_rough_contour = cv2.drawContours(rgb.copy(), [table_detector.round_int32(table_contour)], 0, (255, 0, 255),1)
        else:
            rgb_rough_contour = rgb.copy()
        cv2.imshow("rgb_rough_contour", rgb_rough_contour)
        cv2.waitKey(3)
