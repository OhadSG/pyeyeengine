import glob
import sys
import json
import socket
import cv2
import numpy as np
import base64
import time
from pyeyeengine.server.socket_wrapper import SocketWrapper

# How to use:
# os.system("python3 /usr/local/lib/python3.5/dist-packages/pyeyeengine/send_internal_request.py arg1 arg2");

def open_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.connect(('192.168.0.113', 5006))
    sock.connect(('localhost', 5006))
    MESSAGE_SIZE_INDICATOR_LENGTH = 4
    return SocketWrapper(sock)


if __name__ == '__main__':
    caller = "Dev"#sys.argv[0]
    message_name = "auto_focus"  # sys.argv[1]
    params = {}
    # params = {"focus": 0, "amount": 10}
    # params = '{"screen_height": 800, "screen_width": 1280, "key_points_extractor": "silhouette"}'

    print("{} is sending request named '{}' with params: '{}'".format(caller, message_name, params))

    sw = open_socket()

    # while True:
    sw.send(json.dumps({'name': "{}".format(message_name), 'params': params}))
    message = json.loads(sw.receive_message().decode("utf-8"))
    print("Response: {}".format(message))

        # img_base64 = bytes(message["data"].encode())
        # print(type(img_base64))
        # nparr = np.frombuffer(base64.b64decode(img_base64), np.uint8)
        # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)