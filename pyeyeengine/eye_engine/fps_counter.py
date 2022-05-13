import numpy as np
import time

class FPSCounter:
    def __init__(self, check_every_n_frames=30):
        self.frame_count = 0
        self.fps = 0
        self.check_every_n_frames = int(check_every_n_frames)
        assert(self.check_every_n_frames >= 1)
        self.start_time = time.monotonic()

    def process_frame(self):
        self.frame_count += 1
        if self.frame_count % self.check_every_n_frames == 0:
            if self.frame_count == 0:
                self.start_time = time.monotonic()
            else:
                end_time = time.monotonic()
                self.fps = self.check_every_n_frames/np.maximum((end_time - self.start_time), 0.00001)
                self.start_time = end_time
