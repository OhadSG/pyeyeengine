import json
import os
import threading
import time

from pyeyeengine.eye_engine.eye_engine import EyeEngine
from pyeyeengine.server.socket_wrapper import SocketWrapper
from pyeyeengine.utilities.global_params import EngineType

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_LOG = FILE_PATH + "/../num_key_key_points.txt"


class KeyPointsExtractorSocket:

    def __init__(self) -> None:
        self._client_wrappers = []
        self._engine = None
        self._lock = threading.Lock()
        self.is_active = False

    def add_new_client(self, client):
        self._lock.acquire()
        self._client_wrappers.append(SocketWrapper(client))
        self._lock.release()
        return True

    def remove_all_clients(self):
        self._lock.acquire()
        self._client_wrappers = []
        self._lock.release()

    def set_engine(self, engine: EyeEngine):
        self._lock.acquire()
        self._engine = engine
        self._lock.release()
        return True

    def start(self):
        threading.Thread(target=self._send_key_points_in_loop, daemon=True).start()

    def stop(self):
        self.remove_all_clients()
        time.sleep(5)

    def _send_key_points_in_loop(self):
        try:
            self._try_send_key_points_in_loop()
        except ConnectionError:
            pass

    def _try_send_key_points_in_loop(self):
        while True:
            self._lock.acquire()

            if self._engine.engine_type == EngineType.COM:
                centers_of_mass, pixels, cls, jumps_init, jumps_in_game = self._engine.get_com()
                self._client_wrappers = [socket for socket in self._client_wrappers
                                         if socket.send(
                        json.dumps({'status': 'ok', 'data': (
                            self._parse_com_points_to_dict(centers_of_mass, pixels, cls, jumps_init, jumps_in_game))}))]
            else:
                key_points = self._engine.process_frame()
                self._client_wrappers = [socket for socket in self._client_wrappers
                                         if socket.send(
                        json.dumps({'status': 'ok', 'data': (self._parse_key_points_to_dict(key_points))}))]
            self._lock.release()
            if len(self._client_wrappers) > 0:
                self.is_active = True
            else:
                self.is_active = False
                break

    def _parse_com_points_to_dict(self, centers_of_mass, pixels, cls, jumps_init,jumps_in_game):
        if not isinstance(centers_of_mass, list):
            objects_list = []
            for key in centers_of_mass:
                objects_list.append({"row": pixels[key][0], "col": pixels[key][1], "x": centers_of_mass[key][0], "y": centers_of_mass[key][1], "z": centers_of_mass[key][2], "object_id": int(key)})
            return {"n_objects":len(centers_of_mass), "objects": objects_list, "objects_in": cls.objects_in, "objects_out": cls.objects_out, "unite_candidates": cls.unites, "jumps_init": jumps_init, "jumps_in_game": jumps_in_game}
        else:
            return {"n_objects": 0, "objects": [], "objects_in": [], "objects_out": cls.objects_out, "unite_candidates": cls.unites, "jumps_init": jumps_in_game}


    def _parse_key_points_to_dict(self, key_points):
        return [self._parse_key_point_to_dict(k) for k in key_points]

    def _parse_key_point_to_dict(self, key_point):
        return [{"x": p.tolist()[0], "y": p.tolist()[1], "id": int(key_point.id)} for p in key_point.key_pts]
