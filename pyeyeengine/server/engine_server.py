import json
import socket
import threading
import time

from pyeyeengine.server.request_distributor import RequestDistributor
from pyeyeengine.server.socket_wrapper import SocketWrapper
from pyeyeengine.utilities.rtc_tools import rtc_installed
from pyeyeengine.utilities.logging import Log
from pyeyeengine.utilities.session_manager import *
from pyeyeengine.utilities.metrics import Counter
import logging

logger = logging.getLogger(__name__)

connection_count = Counter(
    'engine_connection_count',
    namespace='pyeye',
)

class EngineServer:
    def __init__(self, request_distributor: RequestDistributor, port=5006) -> None:
        logger.info(
            "Engine Server Start Up",
            extra={
                "session_id": get_session_id(),
                "version": Log.get_engine_version(),
                "serial": Log.get_system_serial(),
                "rtc_installed": rtc_installed(),
            },
        )

        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._request_distributor = request_distributor

    def start(self):
        self._start_serving_loop()

    def _start_serving_loop(self):
        while True:
            try:
                self._try_start_serving_loop()
            except:
                logger.exception("Failed to serve loop. Retrying in 4 seconds")
                time.sleep(4)
            finally:
                self._close_socket()

    def _close_socket(self):
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
            self._sock.close()
        except:
            logger.exception("Failed to close socket")

    def _init_socket(self):
        logger.debug("Binding socket")
        self._sock.bind(("", self._port))
        self._sock.listen(5)
        logger.info("Socket is ready at port {}".format(self._port))

    def _try_start_serving_loop(self):
        self._init_socket()
        while True:
            client, address = self._sock.accept()
            logger.debug("Accepted connection from {}".format(address))
            connection_count.inc()
            threading.Thread(name='EngineServerConnection', target=self._connect_to_client, args=(client,), daemon=True).start()

    def _connect_to_client(self, client):
        client_wrapper = SocketWrapper(client)
        request = client_wrapper.receive_message()
        while request is not None:
            request_json = json.loads(request.decode("utf-8"))
            logging.debug('Received request {}'.format(request))
            request_json['client'] = client
            response = self._request_distributor.distribute(request_json)
            client_wrapper.send(response)
            request = client_wrapper.receive_message()
