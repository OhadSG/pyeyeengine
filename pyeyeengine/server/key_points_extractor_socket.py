import json
import os
import threading
import time

from pyeyeengine.eye_engine.engine_base import EngineBase
from pyeyeengine.server.socket_wrapper import SocketWrapper
from pyeyeengine.utilities.global_params import EngineType
from pyeyeengine.utilities.metrics import Counter, Gauge
from pyeyeengine.camera_utils.frame_manager import FrameManager
import logging

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_LOG = FILE_PATH + "/../num_key_key_points.txt"

logger = logging.getLogger(__name__)

delay = float(os.getenv('PYEYE_POINTS_EXTRACTOR_SAMPLE_INTERVAL', default='0'))
key_point_loop_counter = Counter(
    'key_point_loop_count',
    namespace='pyeye',
)

key_point_sent_counter = Counter(
    'key_points_sent_count',
    namespace='pyeye',
)

key_point_client_count = Gauge(
    'key_point_client_count',
    namespace='pyeye',
)

class KeyPointsExtractorSocket:
    def __init__(self) -> None:
        self._client_wrappers = []
        self._engine = None
        self._lock = threading.Lock()
        self.is_active = False
        self.screen_size = (1280,800)
        self.interactions_counter = [0 for i in range(9)]

    def add_new_client(self, client):
        with self._lock:
            logger.info('Added new client')
            self._client_wrappers.append(SocketWrapper(client))
        return True

    def remove_all_clients(self):
        logger.info('Remove all clients')
        with self._lock:
            self._client_wrappers = []

    def set_engine(self, engine: EngineBase):
        logger.info('Set engine to {}'.format(engine))
        with self._lock:
            self._engine = engine

    def start(self):
        logger.info('Start')
        threading.Thread(
            name='KeyPointExtractorSocketThread',
            target=self._send_key_points_in_loop,
            daemon=True,
        ).start()

    def stop(self):
        logger.info('Stop')
        self.remove_all_clients()
        time.sleep(5)

    def _send_key_points_in_loop(self):
        try:
            self._try_send_key_points_in_loop()
        except ConnectionError:
            pass

    def _try_send_key_points_in_loop(self):
        depth_stream = FrameManager.getInstance().depth_stream
        while True:
            with self._lock:
                key_point_loop_counter.inc({
                    'engine_type': type(self._engine).__name__,
                })

                depth_stream.start()
                depth_stream.wait_for_next_frame()

                if self._engine.engine_type == EngineType.DynamicSurface:
                    surface_data = self._engine.process_frame()
                    message = {'status': 'ok', 'data': surface_data}
                elif self._engine.engine_type == EngineType.COM:
                    centers_of_mass, pixels, cls, jumps_init, jumps_in_game = self._engine.get_com()
                    message = {
                        'status': 'ok',
                        'data': (
                                self._parse_com_points_to_dict(centers_of_mass, pixels, cls, jumps_init, jumps_in_game))
                    }
                else:
                    key_points = self._engine.process_frame()
                    message = {
                        'status': 'ok',
                        'data': (self._parse_key_points_to_dict(key_points))
                    }

                    # logger.debug('key points: {}'.format(key_points))

                key_point_sent_counter.inc_by(len(key_points), {
                    'engine_type': type(self._engine).__name__,
                })

                for socket in self._client_wrappers:
                    socket.send(json.dumps(message))

                self._client_wrappers = [
                    socket
                    for socket in self._client_wrappers
                    if socket.is_closed == False
                ]

            key_point_client_count.set(len(self._client_wrappers), labels={
                'engine_type': type(self._engine).__name__,
            })
            if len(self._client_wrappers) > 0:
                self.is_active = True
            else:
                self.is_active = False

                logger.info('Stopping key point extractor socket because there aren\'t any more clients')

                if self._engine.engine_type == EngineType.DynamicSurface:
                    self._engine.stop_surface_sampling()

                break

            if delay > 0:
                time.sleep(delay)

    def _parse_com_points_to_dict(self, centers_of_mass, pixels, cls, jumps_init,jumps_in_game):
        if not isinstance(centers_of_mass, list):
            objects_list = []
            for key in centers_of_mass:
                objects_list.append({"row": pixels[key][0], "col": pixels[key][1], "x": centers_of_mass[key][0], "y": centers_of_mass[key][1], "z": centers_of_mass[key][2], "object_id": int(key)})
            return {"n_objects":len(centers_of_mass), "objects": objects_list, "objects_in": cls.objects_in, "objects_out": cls.objects_out, "unite_candidates": cls.unites, "jumps_init": jumps_init, "jumps_in_game": jumps_in_game}
        else:
            return {"n_objects": 0, "objects": [], "objects_in": [], "objects_out": cls.objects_out, "unite_candidates": cls.unites, "jumps_init": jumps_in_game}

    def get_interactions(self):
        temp,self.interactions_counter = self.interactions_counter, [0 for i in range(9)]
        return temp

    def _parse_key_points_to_dict(self, key_points):
        current_interactions = [self._parse_key_point_to_dict(k) for k in key_points]
        for interaction in current_interactions:
            for sub_interaction in interaction:
                cell_size_x = self.screen_size[0] // 3
                cell_size_y = self.screen_size[1] // 3
                x = int(sub_interaction["x"]) // cell_size_x
                y = int(sub_interaction["y"]) // cell_size_y
                index = min((3 * 3-1),(x * 3 + y))
                self.interactions_counter[index]+=1
        # if self.last_interaction_object_count != len(current_interactions):
        #     self.interactions += abs(self.last_interaction_object_count-len(current_interactions))
        #     self.last_interaction_object_count = len(current_interactions)
        return current_interactions

    def _parse_key_point_to_dict(self, key_point):
        return [{"x": p.tolist()[0], "y": p.tolist()[1], "id": int(key_point.id)} for p in key_point.key_pts]
