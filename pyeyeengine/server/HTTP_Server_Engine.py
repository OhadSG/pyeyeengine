import json
import threading
import time
import traceback
import os
import logging

from pyeyeengine.server.socket_wrapper import SocketWrapper
from pyeyeengine.utilities.rtc_tools import rtc_installed
from pyeyeengine.utilities.logging import Log
from pyeyeengine.utilities.session_manager import *
from http.server import BaseHTTPRequestHandler,HTTPServer
from functools import partial
from pyeyeengine.utilities.preferences import EnginePreferences
from pyeyeengine.server.request_distributor import RequestDistributor
from pyeyeengine.utilities.helper_functions import *

from pyeyeengine.calibration.auto_calibrator import AutoCalibrator
from pyeyeengine.dynamic_surface.dynamic_surface_engine import DynamicSurfaceEngine
from pyeyeengine.utilities.logging import Log

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_ERROR_LOG = FILE_PATH + "/../error_log.txt"

logger = logging.getLogger(__name__)

class MyServer(BaseHTTPRequestHandler):
    def __init__(self, request_distributor: RequestDistributor, *args, **kwargs):
        self.request_distributor=request_distributor
        super().__init__(*args, **kwargs)
    def do_GET(self):
        Log.i("Received HTTP request GET {}".format(self.path), extra_details={
            "http_path": self.path,
            "http_method": "GET"
        })
        start = time.monotonic()
        data = ""
        if "get_dynamic_surface_snapshot" in self.path:
            data = json.dumps(self.request_distributor.get_dynamic_surface_snapshot(""))
        elif "Command" in self.path:
            command = self.path[self.path.index('?=')+2:]
            print(command)
            if command == "moveleft":
                self.request_distributor.update_offset_x(-1)
            elif command == "moveright":
                self.request_distributor.update_offset_x(1)
            elif command == "moveup":
                self.request_distributor.update_offset_y(-1)
            elif command == "movedown":
                self.request_distributor.update_offset_y(1)
            elif command == "heightdown":
                self.request_distributor.update_height(-5)
            elif command == "heightup":
                self.request_distributor.update_height(5)
            elif command == "scaledown":
                self.request_distributor.update_scale_y(-1)
            elif command == "scaleup":
                self.request_distributor.update_scale_y(1)
            elif command == "scaleright":
                self.request_distributor.update_scale_x(1)
            elif command == "scaleleft":
                self.request_distributor.update_scale_x(-1)
            elif command == "flip":
                self.request_distributor.update_flip()
            elif command == "tiltleft":
                self.request_distributor.update_tilt_x(-5)
            elif command == "tiltright":
                self.request_distributor.update_tilt_x(5)

            json_object = {}
            json_object["offset_x"] = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.X_OFFSET))
            json_object["offset_y"] = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.Y_OFFSET))
            json_object["scale_x"] = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.SCALE_X))
            json_object["scale_y"] = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.SCALE_Y))
            json_object["height"] = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.DYNAMIC_SURFACE_NEAR_PLANE))
            json_object["tilt"] = string_to_int(EnginePreferences.getInstance().get_text(EnginePreferences.TILT))

            data = json.dumps(json_object)

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(data.encode())

        end = time.monotonic()
        Log.i("Sent HTTP response for GET {}".format(self.path), extra_details={
            "duration_sec": end - start,
            "http_path": self.path,
            "http_method": "GET"
        })


class HTTPEngineServer:
    def __init__(self, request_distributor: RequestDistributor, port: int = None):
        if port is None:
            port = int(os.getenv('PYEYE_HTTP_PORT', default='6007'))

        Log.i(
            "Engine Server Start Up",
            extra_details={
                "session_id": get_session_id(),
                "version": Log.get_engine_version(),
                "serial": Log.get_system_serial(),
                "rtc_installed": rtc_installed()
            },
            flow="engine_server"
        )
        self._port = port
        self.request_distributor = request_distributor
        self.server = None

    def start(self):
        handler = partial(MyServer, self.request_distributor)
        self.server = HTTPServer(("0.0.0.0", self._port), handler)
        logger.info("Server started http://{}:{}".format("0.0.0.0", self._port))
        threading.Thread(name='HTTPEngineServer', target=self.server.serve_forever, daemon=True).start()
