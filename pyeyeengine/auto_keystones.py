import glob

import cv2
import json
import socket

from pyeyeengine.auto_keystones.auto_keystones_utils import fix_keystones
from pyeyeengine.calibration.auto_calibrator import AutoCalibrator
from pyeyeengine.camera_utils.camera_reader import CameraReader
from pyeyeengine.projector_controller.projector_controller import ProjectorController
from pyeyeengine.server.socket_wrapper import SocketWrapper


def open_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 5006))
    # sock.connect(('192.168.0.76', 5006))
    MESSAGE_SIZE_INDICATOR_LENGTH = 4
    return SocketWrapper(sock)





if __name__ == '__main__':
    # ProjectorController().turn_on()
    # GRID_IMAGES_SAVE_PATH =  "/usr/local/lib/python3.6/dist-packages/pyeyeengine/calibration/grid_images/*png"
    # screen_res = cv2.imread(glob.glob(GRID_IMAGES_SAVE_PATH)[0]).shape
    ProjectorController().change_vertical_keystone(0) # 20

    sw = open_socket()
    # sw.send(json.dumps({'name': 'calibrate', "params":{'image_pairs': [{"displayed": "", "viewed": ""}]}}))
    sw.send(json.dumps({'name': 'fix_keystones'}))
    print(sw.receive_message())

    cam = CameraReader(display=False, resxy=(640, 480))
    cv2.imshow("rgb", cam.get_rgb())
    cv2.waitKey(0)
    calibrator = AutoCalibrator(screen_resolution=(1920, 1080))
    fix_keystones(cam, calibrator)
    cam.stop()
    hi=5