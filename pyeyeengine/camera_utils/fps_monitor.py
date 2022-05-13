import threading
import time

class FpsMonitor:
    def __init__(self):
        self.fps = 0
        self._lock = threading.Lock()
        self._refresh_rate = 1.0
        self._frame_counter = 0
        self._last_counter_reset = time.monotonic()

    def on_frame_received(self):
        with self._lock:
            self._frame_counter += 1

            time_since_last_reset = time.monotonic() - self._last_counter_reset

            if time_since_last_reset > self._refresh_rate:
                self.fps = int(
                    self._frame_counter / time_since_last_reset
                )
                self._frame_counter = 0
                self._start_time = time.monotonic()