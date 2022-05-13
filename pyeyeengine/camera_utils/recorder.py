from .frame_stream.frame_stream_base import FrameStreamBase
import os.path
import cv2

class Recorder:
    def __init__(self, stream: FrameStreamBase):
        self.stream = stream

    def start(self):
        pass

    def stop(self):
        pass