import cv2
import numpy as np

from pyeyeengine.auto_keystones.auto_keystones_utils import find_rectification_tform
from pyeyeengine.camera_utils.camera_reader import CameraReader


if __name__ == '__main__':
    cam = CameraReader(display=False, resxy=(640, 480))
    keep_running = True
    while keep_running:
        key = cv2.waitKey(1)
        if key == 27:
            keep_running = False

        dmap = cam.get_depth()
        rgb = cam.get_rgb()
        tform = find_rectification_tform(dmap)
        rectefied_rgb = cv2.warpPerspective(rgb, tform, (rgb.shape[1], rgb.shape[0]))
        cv2.imshow("rect_rgb_lin", rectefied_rgb)
        cv2.waitKey(1)

        # rectefied_rgb = rectefy_rgb_with_depth(dmap, rgb)
        # cv2.imshow("rect_rgb_quad", rectefied_rgb)
        # cv2.waitKey(1)

        cv2.imshow("depth", np.uint8(dmap / 10))
        cv2.waitKey(1)
        cv2.imshow("rgb", rgb)
        cv2.waitKey(1)


    ## Release resources
    cv2.destroyAllWindows()
    cam.stop()
