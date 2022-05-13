import numpy as np
from pyeyeengine.utilities.astra_orbbec_conf import AstraOrbbec
from pyeyeengine.utilities.Euclidean_Rotator import EuclideanRotator

class CenterOfMass:

    def __init__(self, x_rotation=0, y_rotation=0):
        self.camera_conf = AstraOrbbec()
        self.centers_of_mass = []
        self.com_in_pixels = []
        self.x_rotation = x_rotation
        self.y_rotation = y_rotation

        self.rotator_x = EuclideanRotator(angle=self.x_rotation, axis='x')
        self.rotator_y = EuclideanRotator(angle=self.y_rotation, axis='y')

    def calculate_center_of_mass(self, clusters):

        self.centers_of_mass = {}
        self.com_in_pixels = {}
        # If there are no clusters, there are also no centers of mass.
        if not clusters:
            return

        # self.centers_of_mass = np.zeros((len(clusters), 3))  # center of mass in euclidean coordinates.
        # self.com_in_pixels = np.zeros((len(clusters), 2))  # center of mass in pixel values, row and col array.
        self.centers_of_mass = {}  # center of mass in euclidean coordinates.
        self.com_in_pixels = {}  # center of mass in pixel values, row and col array.

        # iterate over all clusters
        com_tmp = np.zeros((len(clusters), 3))
        pix_tmp = np.zeros((len(clusters), 2))

        # for key in clusters:
        #     self.centers_of_mass[key] = np.mean(clusters[key]['cluster_points'], axis=0)

        for i, key in enumerate(clusters):
            com_tmp[i, :] = np.mean(clusters[key]['cluster_points'], axis=0)  # The centers of mass are mean values over X,Y,Z - euclidean coordinates.


        ####FOLLOW THE SCRICT ORDER: FIRST Y-ROTATE, THEN X_ROTATE
        y_rotate_backwards = self.rotator_y.perform_3D_rotation(com_tmp)
        x_rotate_backwards = self.rotator_x.perform_3D_rotation(y_rotate_backwards)
        # Perform the backwards transform of euclidean corrdinates to pixel coordinates.
        pix_tmp[:, 0] = np.uint16(np.round(-((x_rotate_backwards[:, 1] * self.camera_conf.f_x) / x_rotate_backwards[:, 2]) + self.camera_conf.c_x))  # row vector
        pix_tmp[:, 1] = np.uint16(np.round(((x_rotate_backwards[:, 0] * self.camera_conf.f_y) / x_rotate_backwards[:, 2]) + self.camera_conf.c_y))  # col vector

        for i, key in enumerate(clusters):
            self.centers_of_mass[key] = com_tmp[i,:]
            self.com_in_pixels[key] = pix_tmp[i, :]