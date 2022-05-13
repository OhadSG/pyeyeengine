import numpy as np
import time
from skimage import measure
import scipy
from scipy.spatial.distance import cdist
import numpy.matlib as mb
import copy
import pandas as pd
from pyeyeengine.utilities.helper_functions import removeSublist
from pyeyeengine.utilities.logging import Log

class PointCluster:
    def __init__(self, quantize_step_size=100, quant_thr = 30, min_value_cluster = 100, min_size_cluster = 4):

        self.quantize_step_size = quantize_step_size
        self.quant_thr = quant_thr
        self.min_value_cluster = min_value_cluster
        self.min_size_cluster = min_size_cluster
        self.min_number_of_points = 150
        self.flag_previous = 0
        self.labels_previous = 0
        #Matrix containing features of previous clusters.
        #Each row --> one cluster
        #col 0 and col1: cluster centers, col 2: cluster id, col 3: cluster sizes
        self.clusters = {}
        self.cluster_neighbors = []
        self.objects_in = []
        self.objects_out = []
        self.id_list_previous = []
        self.unite_timeout = 8000
        self.unites = []
        self.margin_inserting_delay_if_no_object = 10

    # Calculate return values
    def prepare_exit_with_no_cluster(self):
        if self.flag_previous == 1:
            self.clusters = self.calculate_objects_out()
            self.cluster_neighbors = []
            # if self.cluster_neighbors:
            #     Log.d("Exit of object with delay - probable crash")
            #     idx = np.isin(self.cluster_neighbors, np.fromiter(self.clusters.keys(), dtype=int), invert=False)
            #     if len(idx) > 0:
            #         self.cluster_neighbors = self.cluster_neighbors[np.isin(self.cluster_neighbors, np.fromiter(self.clusters.keys(), dtype = int), invert=False)]
            #     else:
            #         self.cluster_neighbors = []

        if not self.clusters:
            self.flag_previous = 0
            self.clusters = {}
            self.cluster_neighbors = []

    def get_cluster_feature(self, feature = 'cluster_centers'):
        values = self.clusters.values()
        if np.isscalar(list(values)[0][feature]):
            feature_array = np.zeros((len(values), 1))
        else:
            feature_array = np.zeros((len(values), 2))

        for i, v in enumerate(values):
            feature_array[i, :] = v[feature]

        return feature_array

    def create_rows_cols_values(self, x_min_value, x_max_value, max_z_value):
        rows = np.ceil(max_z_value / self.quantize_step_size) + 4
        rows = np.uint16(rows)
        cols = np.ceil((x_max_value - x_min_value) / self.quantize_step_size) + 4
        cols = np.uint16(cols)
        return rows,cols

    def create_quantized_index_point_cloud(self, point_cloud, x_min_value):
        point_cloud_ind_quantized = copy.deepcopy(point_cloud)
        point_cloud_ind_quantized = np.transpose(
            np.array([point_cloud_ind_quantized[:, 0], point_cloud_ind_quantized[:, 2]]))

        point_cloud_ind_quantized[:, 1] = np.floor(point_cloud_ind_quantized[:, 1] / self.quantize_step_size)
        point_cloud_ind_quantized[:, 0] = np.floor(
            (point_cloud_ind_quantized[:, 0] - x_min_value) / self.quantize_step_size)

        if point_cloud_ind_quantized.min() < 0:
            point_cloud_ind_quantized[point_cloud_ind_quantized < 0] = 0

        return point_cloud_ind_quantized

    def calculate_unique_ind(self, point_cloud_ind_quantized, cols):
        #calculate the unique indices of "point_cloud_ind_quantized"
        #use the row-vector representation for the unique calculation in order to save computational time.
        ind_1d = point_cloud_ind_quantized[:,0] * cols + point_cloud_ind_quantized[:,1]
        unique_1d, idx_inverse, counts = np.unique(ind_1d, axis=0, return_inverse=True, return_counts=True)
        unique_ind = np.transpose(np.array([np.floor(unique_1d / cols), np.remainder(unique_1d,cols)]))

        # unique_ind, idx_inverse, counts = np.unique(point_cloud_ind_quantized, axis=0, return_inverse=True, return_counts=True)
        idx_inverse = np.reshape(idx_inverse, (-1, 1))
        unique_ind = np.uint16(unique_ind)

        return unique_ind, idx_inverse, counts

    def create_quant_matrix(self, rows, cols, unique_ind, counts):
        quant_matrix = np.zeros((rows, cols))
        inv_idx_matrix = np.zeros((rows, cols))

        q = np.reshape(np.arange(unique_ind.shape[0]), (-1, 1))
        unique_ind = np.hstack((unique_ind, q))

        unique_ind = unique_ind[np.where(unique_ind[:,0] < rows) and np.where(unique_ind[:,1] < cols)]
        quant_matrix[unique_ind[:, 1], unique_ind[:, 0]] = counts
        inv_idx_matrix[unique_ind[:, 1], unique_ind[:, 0]] = unique_ind[:, 2]

        return quant_matrix, inv_idx_matrix

    def filter_labels_calculate_cluster_centers(self, labels, quant_matrix):
        # 1. At least one quant needs to have value >= 80
        # 2. Min size needs to be 3

        cluster_centers = np.zeros((labels.max(), 2))
        shift_vector = np.zeros((labels.max()), dtype=np.uint16)
        decrease_label = 0

        # Filter out clusters which are a) too small by extent (number of quants is too small) or b) have too little points in one quant.
        for i in range(1, labels.max() + 1):
            label = labels == i
            nr = np.sum(label)
            if nr < self.min_size_cluster:
                labels[label] = 0
                shift_vector[i:] += 1  # shift vector is shifted by one, bcs label 1 is in row with index zero.!!!!!!!!!!
                continue

            vals = quant_matrix[label]
            if vals.max() < self.min_value_cluster:
                labels[label] = 0
                shift_vector[i:] += 1  # shift vector is shifted by one, bcs label 1 is in row with index zero.!!!!!!!!!!
                continue

            if shift_vector[i - 1] > 0:
                labels[label] = i - shift_vector[i - 1]

            label_indices = np.argwhere(label)
            count_mat = np.transpose(np.vstack((quant_matrix[label_indices[:, 0], label_indices[:, 1]],
                                                quant_matrix[label_indices[:, 0], label_indices[:, 1]])))
            weighted_indices = label_indices * count_mat
            cluster_centers[i - 1 - shift_vector[i - 1], :] = np.sum(weighted_indices, axis=0) / np.sum(count_mat,
                                                                                                        axis=0)

        cluster_centers = cluster_centers[~(cluster_centers == 0).all(1)]  # delete all zero rows

        return labels, cluster_centers


    def find_neighbors(self,labels2, vicinity = 4):

        self.cluster_neighbors = []
        vicinity = int(vicinity * 100/self.quantize_step_size) #optimized for quant_step size of 100, so when the value changes, the vicinity value changes accordingly.
        h, w = labels2.shape
        for i in range(1, labels2.max() + 1):
            label = labels2 == i
            if label.max() == True:
                rowcol = np.argwhere(label)
                (row_min, row_max, col_min, col_max) = (
                np.min(rowcol[:, 0]), np.max(rowcol[:, 0]), np.min(rowcol[:, 1]), np.max(rowcol[:, 1]))

                object_height = row_max - row_min
                object_width = col_max - col_min

                row_min2 = np.max([0, row_min - vicinity])
                row_max2 = np.min([h, row_max + vicinity])
                col_min2 = np.max([0, col_min - vicinity])
                col_max2 = np.min([w, col_max + vicinity])

                testmat = np.zeros_like(labels2)
                testmat[row_min2:row_max2 + 1, col_min2:col_max2 + 1] = True
                testmat[label] = False

                n = list(pd.unique(np.ndarray.flatten(labels2[np.nonzero(testmat & labels2)])))

                if len(n) > 0:
                    n.append(i)
                    self.cluster_neighbors.append(n)


                self.cluster_neighbors = removeSublist(self.cluster_neighbors)

    def check_for_unification(self, dist_mat, quant_matrix, labels):

        id_list = np.fromiter(self.clusters.keys(), dtype = int)
        unite_dict = {}
        unite_cand_ixs = []
        for nbrs in self.cluster_neighbors:

            both = set(nbrs).intersection(id_list)
            idx = np.array([list(id_list).index(x) for x in both])  # idx in self.clusters
            closest_mapping = np.argmin(dist_mat[:, idx], axis=0)  # closest matching in the current clustering

            unique_mapping, counts = np.unique(closest_mapping, axis=0, return_counts=True)
            if counts.max() > 1:
                for i, u in enumerate(unique_mapping):
                    if counts[i] > 1:

                        unite_cand_ix = np.transpose(idx[np.argwhere(closest_mapping == u)])
                        ###check if cluster size has almost doubled:
                        cluster_sizes_previous = self.get_cluster_feature(feature='cluster_size')
                        unite_cand_sizes_previous = cluster_sizes_previous[unite_cand_ix]
                        cluster_cand_size = np.sum(quant_matrix[labels == (u + 1)])
                        #hardcoded, please change.
                        cond = (cluster_cand_size > 0.75 * np.sum(unite_cand_sizes_previous)) and (
                                    cluster_cand_size > 1.2 * np.max(unite_cand_sizes_previous))

                        if cond:
                            unite_dict[u+1] = (id_list[unite_cand_ix]).tolist()[0]
                            unite_cand_ixs.append(unite_cand_ix)

        return unite_dict, unite_cand_ixs

    def calculate_objects_in(self, new_ids=[]):
        self.objects_in = new_ids
        self.objects_in = [int(item) for item in self.objects_in]

    def calculate_objects_out(self, transform_dict={}):
        ids = np.array([item for sublist in transform_dict.values() for item in sublist])
        ids_old = np.fromiter(self.clusters.keys(), dtype=int)
        oo_candidates = list(ids_old[np.isin(ids_old, ids, invert=True)])
        self.objects_out = []
        clusters_tmp = {}

        for id in oo_candidates:
            if self.clusters[id]['oof_counter'] < self.margin_inserting_delay_if_no_object:
                clusters_tmp[id] = self.keep_cluster_values(id, oof_counter_inc=1)
            else:
                self.objects_out.append(int(id))

        return clusters_tmp
        # self.objects_out = list(ids_old[np.isin(ids_old, ids, invert=True)])
        # self.objects_out = [int(item) for item in self.objects_out]

    def keep_cluster_values(self,id, unite_counter_inc = 0, oof_counter_inc = 0):
        cls = {}
        cls['cluster_centers'] = self.clusters[id]['cluster_centers']
        cls['cluster_size'] = self.clusters[id]['cluster_size']
        cls['cluster_points'] = self.clusters[id]['cluster_points']
        cls['unite_counter'] = self.clusters[id]['unite_counter'] + unite_counter_inc
        cls['oof_counter'] = self.clusters[id]['oof_counter'] + oof_counter_inc
        return cls

    def iterate_dist_mat(self, dist_mat, unite_dict, unite_dict_ixs):
        transform_dict = copy.deepcopy(unite_dict)
        all_ixs = np.array([elem for singleList in unite_dict_ixs for elem in singleList])

        if unite_dict:
            keys = np.fromiter(unite_dict.keys(), dtype=int)
            # vals = unite_dict.values()
            # vals = np.array([item for sublist in vals for item in sublist])
            dist_mat[keys-1, :] = 1e7
            dist_mat[:, all_ixs] = 1e7 - 1

        while np.min(dist_mat) < 1e7 - 1:
            argmin = dist_mat.argmin()
            row, col = argmin // dist_mat.shape[1], argmin % dist_mat.shape[1]
            transform_dict[row + 1] = [self.id_list_previous[col]]
            dist_mat[row, :] = 1e7
            dist_mat[:, col] = 1e7

        return transform_dict

    def assign_new_cluster_ids(self, transform_dict, labels):

        labels_assigned = np.fromiter(transform_dict.keys(), dtype=int)
        labels_total = np.arange(1, labels.max() + 1)
        labels_unassigned = labels_total[np.isin(labels_total, labels_assigned, invert=True)]

        ids_assigned = np.array([item for sublist in transform_dict.values() for item in sublist])

        if labels_unassigned.shape[0] > 0:
            ids_possible = np.arange(1, ids_assigned.max() + labels_unassigned.shape[0] + 1)
            ids_possible = ids_possible[np.isin(ids_possible, ids_assigned, invert=True)]

        new_ids = []
        for i, l_ua in enumerate(labels_unassigned):
            transform_dict[l_ua] = [ids_possible[i]]
            new_ids.append(ids_possible[i])

        return transform_dict, new_ids

    def update_clusters_dict(self, cluster_center, quant_matrix, label):
        cluster_dict = {}
        cluster_dict['cluster_centers'] = cluster_center
        cluster_dict['cluster_size'] = np.sum(quant_matrix[label])
        cluster_dict['unite_counter'] = 0
        cluster_dict['oof_counter'] = 0

        return cluster_dict


    def apply_clustering(self, point_cloud, x_max_value, x_min_value, max_z_value):

        self.id_list_previous = list(self.clusters.keys())
        self.objects_in = []
        self.objects_out = []
        self.unites = []

        if point_cloud.shape[0] < self.min_number_of_points:
            self.prepare_exit_with_no_cluster()
            return


        #rows and cols of quant matrix
        rows, cols = self.create_rows_cols_values(x_min_value, x_max_value, max_z_value)

        ################################################################################################################

        #Create matrix that contains the point cloud in the quantized representation
        point_cloud_ind_quantized = self.create_quantized_index_point_cloud(point_cloud, x_min_value)

        ################################################################################################################
        #Delete entries which are outside of the quantization possible values.
        false_rows = point_cloud_ind_quantized[:,0] >= cols
        false_cols = point_cloud_ind_quantized[:,1] >= rows
        #########################
        correct_idx = np.invert(np.logical_or(false_cols, false_rows))

        point_cloud_ind_quantized = point_cloud_ind_quantized[correct_idx, :]

        if point_cloud_ind_quantized.shape[0] < self.min_number_of_points:
            self.prepare_exit_with_no_cluster()
            return

        point_cloud = point_cloud[correct_idx, :]

        ################################################################################################################
        unique_ind, idx_inverse, counts = self.calculate_unique_ind(point_cloud_ind_quantized, cols)

        #Create quant_matrix and inv_idx_matrix
        #quant_matrix: entries contain the number of points within a certain quantization area.
        #inv_idx_matrix: contains the map of the unique indices in the quant_matrix to the specified index value in unique_ind[:,2].
        quant_matrix, inv_idx_matrix = self.create_quant_matrix(rows, cols, unique_ind, counts)

        quant_matrix_thr = np.where(quant_matrix > self.quant_thr, 1, 0)
        #labels: contains the connected components (i.e. objects) of the quant_matrix. Minimum count number is assumed to be > than the value in self.quant_thr.
        labels = measure.label(quant_matrix_thr)
        #############################################################################################
        #Filter out to small "clusters" in labels and calculate the cluster centers out of labels.
        labels, cluster_centers = self.filter_labels_calculate_cluster_centers(labels, quant_matrix)

        #If no labels are left, exit the function as no clusters are present.
        if labels.max() == 0:
            self.prepare_exit_with_no_cluster()
            return

        # if (self.flag_previous == 1) and (cluster_centers.shape[0] > 1):
        if (self.flag_previous == 1):
            centers_previous = self.get_cluster_feature(feature = 'cluster_centers')
            dist_mat = cdist(cluster_centers, centers_previous)

            #########Check for unification of clusters
            unite_dict, unite_dict_ix = self.check_for_unification(dist_mat, quant_matrix, labels)
            transform_dict = self.iterate_dist_mat(dist_mat, unite_dict, unite_dict_ix)
            transform_dict, new_ids = self.assign_new_cluster_ids(transform_dict, labels)

            #switch labels
            labels2 = np.zeros_like(labels)
            clusters_tmp = {}

            for i in range(1, labels.max() + 1):
                label = labels == i
                labels2[label] = transform_dict[i][0]

                ids = transform_dict[i]
                if len(ids) > 1:
                    cntrs = np.fromiter(map(lambda k: self.clusters[k]['unite_counter'], tuple(ids)), dtype=np.int)

                    if (cntrs >= self.unite_timeout).any():
                        del transform_dict[i]
                        continue

                    for id in ids:
                        #Keep features from previous frame
                        clusters_tmp[id] = self.keep_cluster_values(id,unite_counter_inc=1)
                        # clusters_tmp[id]['cluster_centers'] = self.clusters[id]['cluster_centers']
                        # clusters_tmp[id]['cluster_size'] = self.clusters[id]['cluster_size']
                        # clusters_tmp[id]['cluster_points'] = self.clusters[id]['cluster_points']
                        # clusters_tmp[id]['unite_counter'] = self.clusters[id]['unite_counter'] + 1
                else:
                    #Update features
                    ids = ids[0]
                    clusters_tmp[ids] = self.update_clusters_dict(cluster_centers[i - 1, :], quant_matrix, label)

            if not transform_dict:
                self.prepare_exit_with_no_cluster()
                return

            self.calculate_objects_in(new_ids)
            clusters_tmp.update(self.calculate_objects_out(transform_dict))

            #find neighbors
            if cluster_centers.shape[0] > 1:
                self.find_neighbors(labels2, vicinity = 4)

            self.clusters = clusters_tmp
            self.unites = []
            for key in unite_dict:
                self.unites.append(unite_dict[key])
        else:
            clusters_tmp = {}
            unite_dict = {}
            for i in range(1,labels.max()+1):
                clusters_tmp[i] = self.update_clusters_dict(cluster_centers[i - 1, :], quant_matrix, labels == i)

            self.clusters = clusters_tmp
            label_range = np.arange(1, cluster_centers.shape[0] + 1)

            self.calculate_objects_in(list(label_range))
            labels2 = labels

            if labels.max() > 1:
                self.find_neighbors(labels2, vicinity = 4)

        self.flag_previous = 1

        #Get all clusters where a "unification" has occured.
        unite_vals = unite_dict.values()
        unite_vals = np.array([item for sublist in unite_vals for item in sublist])

        for i in range(1, labels2.max() + 1):

            if i in unite_vals:
                continue

            label = labels2 == i

            idx = inv_idx_matrix[label]
            idx = np.uint16(idx)
            idx_in_pointcloud = (idx_inverse == idx).any(axis=1)

            cluster_points = point_cloud[idx_in_pointcloud, :]
            if cluster_points.shape[0] == 0:
                continue

            self.clusters[i]['cluster_points'] = cluster_points
            # cluster_dict = {'cluster_points': cluster_points, 'cluster_id': i}
