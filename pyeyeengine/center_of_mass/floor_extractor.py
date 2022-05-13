import numpy as np
from pyeyeengine.utilities.logging import Log
import copy
from sklearn.cluster import DBSCAN
from pyeyeengine.utilities.helper_functions import get_best_plane_from_points

class FloorExtractor():

    def __init__(self, point_cloud, alg_conf, percentage=0.2):
        self.calculate_floor_normal(point_cloud=point_cloud, alg_conf=alg_conf, percentage = percentage)

    def calculate_low_points_flat(self, point_cloud, percentage):
        #get the #percentage - % of lowest poiunts according to the y-dimension
        points = copy.deepcopy(point_cloud)
        points = points[points[:, 1].argsort(), :]

        nr = int(np.round(percentage * points.shape[0]))
        low_points = points[:int(nr)]

        #oppress the x-dimension.
        low_points_flat = np.transpose(np.array([low_points[:, 1], low_points[:, 2]]))

        return low_points, low_points_flat

    def perform_dbscan_filtering(self, alg_conf, low_points_flat):

        # perform dbscan for outlier removal
        alg_conf = {'eps': 150, 'min_samples': 100} #Values hardcoded, as they have been found empirically.
        dbscan = DBSCAN(eps=alg_conf['eps'], min_samples=alg_conf['min_samples'])
        dbscan.fit(low_points_flat)

        # Get the biggest cluster from the dbscan algorithm, assume as floor

        labels = dbscan.labels_
        unique_labels = set(labels)

        if -1 in unique_labels:
            unique_labels.remove(-1)  # remove the noisy cluster

        return labels, unique_labels

    def determine_floor_points(self, low_points, labels, unique_labels):
        cluster_sizes = []
        cluster_points = []

        # low_points_filtered contain the points which are assumed to belong to the floor.

        if len(unique_labels) > 0:
            for l in unique_labels:
                class_member_mask = (labels == l)
                cluster_points.append(low_points[class_member_mask, :])
                cluster_sizes.append(np.sum(class_member_mask))

            if len(unique_labels) > 1:
                new_order = np.array(cluster_sizes).argsort()
                low_points_filtered = cluster_points[new_order[-1]]
            else:
                low_points_filtered = cluster_points[0]

            return low_points_filtered

        else:
            Log.w("No clusters found for floor extraction")
            return None


    def calculate_floor_normal(self, point_cloud, alg_conf, percentage = 0.2):

        low_points, low_points_flat = self.calculate_low_points_flat(point_cloud=point_cloud, percentage=percentage)
        labels, unique_labels = self.perform_dbscan_filtering(alg_conf, low_points_flat)
        low_points_filtered = self.determine_floor_points(low_points, labels, unique_labels)
        if low_points_filtered is not None:
            normal, distance, _ = get_best_plane_from_points(point_cloud)
            self.floor_normal_coord = normal
            self.floor_normal_coord.append(distance)
        else:
            self.floor_normal_coord = [0, 0, 0, 0]



