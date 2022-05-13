import cv2
from pyeyeengine.utilities.Scripts.camera_scripts.LocalFrameManager import LocalFrameManager

if __name__ == '__main__':
    frame_manager = LocalFrameManager()

    while True:
        depth = frame_manager.get_depth_frame()
        rgb = frame_manager.get_rgb_frame()

        cv2.imshow("rgb", rgb)
        cv2.imshow("depth", depth)
        cv2.waitKey(1)