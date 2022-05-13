import json
import socket
import threading
import time
import traceback

from pyeyeengine.server.request_distributor import RequestDistributor
from pyeyeengine.server.socket_wrapper import SocketWrapper
from pyeyeengine.utilities.logging import Log

class EngineServer:

    def __init__(self, port=5006) -> None:
        Log.i("[SERVER] Starting up...")
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._request_distributor = RequestDistributor()

    def start(self):
        self._start_serving_loop()

    def _start_serving_loop(self):
        while True:
            try:
                self._try_start_serving_loop()
            except Exception as e:
                Log.e("[SERVER] Failed to serve loop. Retrying in 4 seconds", extra_details={"error":"{}".format(e)})
                time.sleep(4)
            finally:
                self._close_socket()

    def _close_socket(self):
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
            self._sock.close()
        except Exception as e:
            Log.e("[SERVER] Failed to close socket", extra_details={"error":"{}".format(e)})

    def _init_socket(self):
        Log.d("[SERVER] Binding socket")
        self._sock.bind(("", self._port))
        self._sock.listen(5)
        Log.i("[SERVER] Socket is ready", extra_details={"port":self._port})

    def _try_start_serving_loop(self):
        self._init_socket()
        while True:
            client, address = self._sock.accept()
            Log.d("[SERVER] Accepted connection from {}".format(address))
            threading.Thread(target=self._connect_to_client, args=(client,), daemon=True).start()

    def _connect_to_client(self, client):
        client_wrapper = SocketWrapper(client)
        request = client_wrapper.receive_message()
        while request is not None:
            request_json = json.loads(request.decode("utf-8"))
            request_json['client'] = client
            response = self._request_distributor.distribute(request_json)
            client_wrapper.send(response)
            request = client_wrapper.receive_message()
