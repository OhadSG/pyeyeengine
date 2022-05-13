
from threading import Timer
import sys

from pyeyeengine.camera_utils.camera_reader import CameraReader


class check_cam_works:
    def __init__(self):
        self.keep_running = True

        try:
            cam = CameraReader()
        except:
            print(1)
            sys.exit(0)

        timer = Timer(3, self.on_timer)
        timer.start()

        self.frame_num = 0
        # create engine

        while self.keep_running:
            depth_map = cam.get_depth()
            rgb = cam.get_rgb()
            if depth_map.sum()>0:
                self.frame_num += 1

        cam.stop()

    def on_timer(self):
        self.keep_running = False
        if self.frame_num < 30:
            print(0)
        else:
            print(1)



if __name__ == "__main__":
    check_cam_works()


