import glob
import time

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
    #sock.connect(('192.168.0.35', 5006))
    MESSAGE_SIZE_INDICATOR_LENGTH = 4
    return SocketWrapper(sock)





if __name__ == '__main__':
    start_time = time.clock()
    sw = open_socket()
    print("connect to socket: %f" % (time.clock() - start_time))
    # sw.send(json.dumps({'name': "reload_table_data"}))
    sw.send(json.dumps({'name': "get_center_of_mass"}))
    # sw.send(json.dumps({'name': 'calibrate', "params":{'image_pairs': [{"displayed": "", "viewed": ""}]}}))
    # sw.send(json.dumps({'name': 'calibrate_with_image_server',
    #                     'params' : {"screen_width": 1280, "screen_height": 800, "mode": "table"}}))
    # sw.send(json.dumps({'name': "reset_saved_data"}))
    message = sw.receive_message()

    while (True):
        print('time for processing 2: %f' % (time.time() - start_time) )
        start_time = time.time()
        print('before')
        message = sw.receive_message()
        print("time for processing 1: %f" % (time.time() - start_time))
        print(message)
    hi=5

