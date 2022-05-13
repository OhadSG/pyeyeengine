import numpy as np


class EuclideanRotator:

    def __init__(self, angle, axis='x'):
        self.angle = angle
        self.axis = axis
        self.sin_a = np.sin(np.deg2rad(self.angle))
        self.cos_a = np.cos(np.deg2rad(self.angle))
        if self.axis == 'x':
            self.rotation_matrix = self.rotate_x()
        elif self.axis == 'y':
            self.rotation_matrix = self.rotate_y()
        elif self.axis == 'z':
            self.rotation_matrix = self.rotate_z()

    def perform_3D_rotation(self, point_matrix):
        return np.transpose(np.matmul(self.rotation_matrix, np.transpose(point_matrix)))

    def rotate_x(self):
        rotation_matrix = np.array(([1, 0, 0], [0, self.cos_a, -self.sin_a], [0, self.sin_a, self.cos_a]))
        return rotation_matrix

    def rotate_y(self):
        rotation_matrix = np.array(([self.cos_a, 0, self.sin_a], [0, 1, 0], [-self.sin_a, 0, self.cos_a]))
        return rotation_matrix

    def rotate_z(self):
        rotation_matrix = np.array(([self.cos_a, -self.sin_a, 0], [self.sin_a, self.cos_a, 0], [0, 0, 1]))
        return rotation_matrix
