import json
import socket
import base64
import numpy as np
from PIL import Image

MESSAGE_SIZE_INDICATOR_LENGTH = 4

class SocketWrapper:
    def __init__(self, socket) -> None:
        self._socket = socket

    def send(self, message_bytes):
        try:
            message_length = len(message_bytes)
            # print("\n1. request sent: {}".format(message_length.to_bytes(MESSAGE_SIZE_INDICATOR_LENGTH, byteorder='big') + message_bytes.encode()))
            self._socket.send(message_length.to_bytes(MESSAGE_SIZE_INDICATOR_LENGTH, byteorder='big') + message_bytes.encode())
            return True
        except:
            return False

    def receive_message(self):
        try:
            return self._try_receive_message()
        except ConnectionResetError:
            return None

    def _try_receive_message(self):
        # print("\n2. reading message length")
        request_length = self._read_message_length()
        # print("\n3. reading message")
        return self._read_n_bytes(request_length)

    def _read_message_length(self):
        return int.from_bytes(self._read_n_bytes(MESSAGE_SIZE_INDICATOR_LENGTH), byteorder='big', signed=True)

    def _read_n_bytes(self, n):
        data = b''
        while len(data) < n:
            # print("\n  - data length: {}".format(n - len(data)))
            packet = self._socket.recv(n - len(data))
            if not packet:
                raise ConnectionResetError
            # print("\n  - packet: {}".format(packet))
            data += packet
        # print("\n  - final data: {}".format(data))
        return data

def open_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('192.168.0.36', 5006))
    MESSAGE_SIZE_INDICATOR_LENGTH = 4
    return SocketWrapper(sock)

if __name__ == '__main__':
    # message_name = "get_dynamic_surface_snapshot"
    message_name = "get_rbg_as_png"

    params = {}
    # params = {"focus": 0, "amount": 10}
    # params = {"screen_height": 800, "screen_width": 1280, "key_points_extractor": "silhouette"}
    # params = {"screen_width": 1280, "mode": "table", "screen_height": 800}

    sw = open_socket()
    # sw.send(json.dumps({'name': "{}".format(message_name), 'params': params}))
    sw.send(json.dumps({'name': "{}".format(message_name)}))
    # counter = 0

    while True:
    # counter < 1:
        message = json.loads(sw.receive_message().decode("utf-8"))
        # print("\n4. received response: \n{}\n".format(message))
        # counter += 1

        img_base64 = bytes(message["data"].encode())
        # print(type(img_base64))
        nparr = np.frombuffer(base64.b64decode(img_base64), np.uint8)
        # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)