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

    sw = open_socket()
    # sw.send(json.dumps({'name': 'calibrate', "params":{'image_pairs': [{"displayed": "", "viewed": ""}]}}))
    sw.send(json.dumps({'name': 'get_fps'}))
    print(sw.receive_message())

