class AstraOrbbec:

    def __init__(self, scale_factor=2):

        # Astra Orbbec intrinsic camera parameters
        # self.f_x = 585.191 / scale_factor
        # self.f_y = 543.456 / scale_factor
        # self.c_x = 316.568 / scale_factor
        # self.c_y = 257.15 / scale_factor
        self.f_x = 533.9336430951793 / scale_factor
        self.f_y = 537.9956934362506 / scale_factor
        self.c_x = 295.83182676051626 / scale_factor
        self.c_y = 248.5586495610084 / scale_factor

        self.x_res = int(480 / scale_factor)
        self.y_res = int(640 / scale_factor)
        self.z_res = 2 ** 16