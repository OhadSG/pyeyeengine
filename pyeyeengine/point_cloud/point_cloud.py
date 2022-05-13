import numpy as np
import copy
from pyeyeengine.utilities.astra_orbbec_conf import AstraOrbbec
from pyeyeengine.utilities.Euclidean_Rotator import EuclideanRotator
from pyeyeengine.utilities.logging import Log

class PointCloud:

    def __init__(self, depth_image, x_rotation = 0, y_rotation = 0, camera_conf=None):
        self.camera_conf = camera_conf if camera_conf is not None else AstraOrbbec()

        self.x_rotation = x_rotation
        self.y_rotation = y_rotation
        self.rotator_x = EuclideanRotator(angle=self.x_rotation, axis='x')
        self.rotator_y = EuclideanRotator(angle=self.y_rotation, axis='y')

        self.point_cloud = self.calculate_point_cloud(depth_image)

    def calculate_point_cloud(self, depth_image):
        rows, cols = depth_image.shape

        # calculate a matrix containing row vector, col vector, Z values, X values and Y values.
        help_matrix = np.transpose(np.array([np.matrix.flatten(np.tile(np.arange(rows), (cols, 1)), order='F'),
                                 np.matrix.flatten(np.tile(np.arange(cols), (rows, 1)), order='C'),
                                 np.matrix.flatten(depth_image, order='C')]))

        help_matrix = help_matrix[~(help_matrix[:, 2] == 0), :]

        # calc X and Y values: X and Y are switched!!!!
        Y = -((help_matrix[:, 0] - self.camera_conf.c_x) * help_matrix[:, 2]) / self.camera_conf.f_x  # Y values
        X = ((help_matrix[:, 1] - self.camera_conf.c_y) * help_matrix[:, 2]) / self.camera_conf.f_y  # X values

        # Contains X, Y and Z coordinates in the specified order.
        tmp = np.transpose(np.array([X,Y,help_matrix[:, 2]]))

        #Add possible point cloud rotations
        if self.x_rotation != 0:
            tmp_rotate_x = self.rotator_x.perform_3D_rotation(tmp)
        else:
            tmp_rotate_x = tmp

        if self.y_rotation != 0:
            tmp_rotate_xy = self.rotator_y.perform_3D_rotation(tmp_rotate_x)
        else:
            tmp_rotate_xy = tmp_rotate_x

        return tmp_rotate_xy

    def update_point_cloud(self, depth_image):
        self.point_cloud = self.calculate_point_cloud(depth_image)

    def filter_by_function(self, pol_approx, axis = 0, margin = 100): #0 for X, 1 for Y
        if ((axis == 0) or (axis == 1)):
             valid_idx = np.where((pol_approx(self.point_cloud[:,axis]) > self.point_cloud[:,2] + margin), True, False)
             self.point_cloud = self.point_cloud[valid_idx, :]

        else:
            Log.w('Wrong axis specified')

    def filter_by_lookup(self, lookup, x_min, x_max, step_size=20, margin=100):

        # Currently implemented for X-axis as index values (all the points are considered according to their x-z distribution while ignoring y) --> point_cloud[:,0].
        idx_vector = (step_size * np.round(self.point_cloud[:,
                                           0] / step_size))  # X-Values are rounded so that they have the same resolution as the background function (lookup-table).
        idx_vector = np.array((idx_vector - x_min) / step_size)
        idx_vector = np.clip(idx_vector, 0, (x_max - x_min) / step_size)
        idx_vector = idx_vector.astype(np.int16)

        # Now, idx-vector values are in range 0,...,N which correspond exactly to the possible index values of the background lookup-table.
        # A direct comparison of the z-values is now possible.

        # Valid points are those that have smaller z-values + margin (for a specific x-value) than the corrsponding z-values of the
        # background lookup-table. Otherwise points are considered as background.
        # valid_idx = np.where((lookup[idx_vector] > self.point_cloud[:, 2] + margin), True, False)
        # self.point_cloud = self.point_cloud[valid_idx, :]
        self.point_cloud = self.point_cloud[lookup[idx_vector] > self.point_cloud[:, 2] + margin, :]

    #Filter by y-values of 3D-plane
    def filter_by_function_3D(self, coord, margin=200):
        y = (coord[0] * self.point_cloud[:, 0] + coord[2] * self.point_cloud[:, 2] + coord[3]) / (-coord[1])
        # valid_idx = np.where(self.point_cloud[:, 1] > y + margin, True, False)
        # self.point_cloud = self.point_cloud[valid_idx, :]
        self.point_cloud = self.point_cloud[self.point_cloud[:, 1] > y + margin, :]