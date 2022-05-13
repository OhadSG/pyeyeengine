class EngineBase:
    def process_frame(self, display=False, show_time=False, debug=False):
        raise NotImplementedError()

    def display_contours_on_rbg(self, contours, key_pts_list, img, calibrator):
        raise NotImplementedError()

    @property
    def engine_type(self):
        raise NotImplementedError()