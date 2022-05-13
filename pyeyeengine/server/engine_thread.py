import threading

from pyeyeengine.calibration.auto_calibrator import AutoCalibrator
from pyeyeengine.camera_utils.camera_reader import CameraReader
from pyeyeengine.eye_engine.eye_engine import EyeEngine
from pyeyeengine.object_detection.key_points_extractor import FootEdgeExtractor


class EngineThread:

    def __init__(self) -> None:
        self._hands = []
        self._hand_lock = threading.Lock()
        self._should_resume = True
        self._should_resume_lock = threading.Lock()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._set_should_resume(True)
        camera = CameraReader()
        engine = EyeEngine(camera, key_pts_extractor=FootEdgeExtractor(), calibrator=AutoCalibrator(screen_resolution=(0, 0)))
        self._thread = threading.Thread(name='EngineThread', target=self._engine_loop, args=(camera, engine))
        self._thread.daemon = True
        self._thread.start()

    def _engine_loop(self, camera, engine):
        while self._get_should_resume():
            hands = engine.process_frame()
            self._set_hands(hands)
        camera.stop()

    def _set_hands(self, hands):
        with self._hand_lock:
            self._hands = hands

    def read_hands(self):
        with self._hand_lock:
            return [h for h in self._hands]

    def stop(self):
        if self._thread is not None:
            self._set_should_resume(False)
        self._thread = None

    def _set_should_resume(self, value):
        with self._should_resume_lock:
            self._should_resume = value

    def _get_should_resume(self):
        with self._should_resume_lock:
            return self._should_resume
