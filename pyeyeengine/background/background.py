import copy
import numpy as np
import collections
import time
from pyeyeengine.utilities.helper_functions import cart2pol

class BackgroundFunction:

    def __init__(self, point_cloud, step_size=25, padding = 5):
        self.background_dict = {}
        self.step_size = step_size
        self.padding = padding
        self.tmp_background = {}
        self.init_bg_removal_function(point_cloud)

    def init_bg_removal_function(self, point_cloud): #point_matrix columns: X-vals, Y-vals, Z-vals.
        #Set values of the tmp_background dict from scratch withouit taking into account possible previous values of the dict.
        if point_cloud.shape[0] < 10:
            return

        point_matrix = copy.deepcopy(point_cloud)
        point_matrix[:,0] = (self.step_size * np.round(point_matrix[:,0] / self.step_size))  # Quantize x-values by stepsize

        # calculate unique x-values conditioned on minimal z. Returns index values of true entries.
        point_matrix = point_matrix[point_matrix[:, 0].argsort(), :]
        _, indicesList = np.unique(point_matrix[:, 0], return_index=True)

        idx_arr = np.zeros(point_matrix.shape[0], dtype=bool)
        for i in range(1, indicesList.shape[0]):
            idx = np.argmin(point_matrix[indicesList[i - 1]: indicesList[i], 2]) #For each x-value in the range of "step_size", find the z-value with the lowest value (closest point to the camera).
            idx_arr[idx + indicesList[i - 1]] = True

        point_matrix = point_matrix[idx_arr, :] #Filter points

        if point_matrix.shape[0] < 10:
            return

        #Now, for each x-value there is exactly one corresponding z-value. Distance between x-values is exactly the step_size (the quantization factor).
        x_vals = point_matrix[:,0]
        z_vals = point_matrix[:, 2]

        #Pad x- and z-values on both sides in order to enlarge the view of the background function.
        x_pad_left = np.arange( x_vals[0] - self.padding * self.step_size, x_vals[0], self.step_size)
        x_pad_right = np.arange(x_vals[-1] + self.step_size, x_vals[-1] + (self.padding +1) * self.step_size, self.step_size)
        x_padded = np.hstack((x_pad_left, x_vals, x_pad_right))

        z_pad_left = np.ones(self.padding) * z_vals[0]
        z_pad_right = np.ones(self.padding) * z_vals[-1]
        z_padded = np.hstack((z_pad_left , z_vals , z_pad_right))

        #For mode "lookup" calculate the lookup function.
        self.tmp_background['x_min'] = x_pad_left[0]
        self.tmp_background['x_max'] = x_pad_right[-1]

        #The lookup-table needs to be complete. Therefore, for each x-value between x_min and x_max (with the
        #quantization specified in step_size), there needs to be a corresponding z-value. If there were no points for
        #a specific x-quantum, just interpolate the z-function by sample-and-hold (take the z-value to the left and
        #add it to the lookup function). Missing_idx indicates the indexes, for which no valid z-value has been found
        #from the point-cloud.
        list_x = list(x_padded.astype(int))
        list_z = list(z_padded)
        idx = np.arange(list_x[0], list_x[-1]+self.step_size, self.step_size)
        listx_set = set(list_x)

        missing_idx = []

        for i in idx:
            if i not in listx_set:
                missing_idx.append(i)

        for mi in missing_idx:
            new_idx = int((mi - self.tmp_background['x_min']) / self.step_size)
            #list_z.insert(new_idx, list_z[new_idx-1])
            list_z.insert(new_idx, 1e7)

        self.tmp_background['lookup_table'] = np.array(list_z)

    def update_background_function(self):
        #Update parameters. The temporary dict values are written to the actual background dict. This "officially" changes the values of the background function. (--> This dict is used by functions/scripts from outside this class here.)
        self.background_dict['x_min'] = self.tmp_background['x_min']
        self.background_dict['x_max'] = self.tmp_background['x_max']
        self.background_dict['lookup_table'] = self.tmp_background['lookup_table']

    def deepcopy_params(self):
        old_x_min = copy.deepcopy(self.tmp_background['x_min'])
        old_x_max = copy.deepcopy(self.tmp_background['x_max'])
        old_lookup = copy.deepcopy(self.tmp_background['lookup_table'])

        return old_x_min, old_x_max, old_lookup

    def find_occlusion_vector(self, clusters):

        #Check if a cluster occludes points in the background in the x-z-space. Take the leftmost/rightmost point of
        #each cluster and place a line which connects this point to (0/0). "Steepness" indicates the slope of the the
        #function in the x-z-space. Steepness_min is the slope of the line through the leftmost point, steepness_max is
        #the corresponding value for the rightmost point of the cluster.
        steepness_max = []
        steepness_min = []

        for key in clusters:
            points = clusters[key]['cluster_points']
            if points.shape[0] == 0:
                continue
            rho, phi = cart2pol(points[:, 0], points[:, 2]) #Use polar instead of cartesian coordinates.
            # max_coord = np.argmax(points[:,0], axis=0)
            # min_coord = np.argmin(points[:,0], axis=0)
            max_coord = np.argmin(phi, axis=0)    #max = min: this has been changed on purpose!
            min_coord = np.argmax(phi, axis=0)

            steepness_max.append(points[max_coord, 2] / points[max_coord, 0])
            steepness_min.append(points[min_coord, 2] / points[min_coord, 0])

        return steepness_max, steepness_min

    def calculate_block_vector_for_occlusion(self, old_lookup, old_x_min, old_x_max, clusters):

        block_vector = np.zeros_like(old_lookup, dtype=bool)
        steepness_right, steepness_left = self.find_occlusion_vector(clusters)
        x_tmp = np.arange(old_x_min, old_x_max + self.step_size, self.step_size)

        for i in range(len(steepness_right)):
            # Calculate the lines which go out from the leftmost/rightmost point of the cluster. Everything between the
            # two lines is assumed to be occluded by the cluster so the background function should not get updated
            # in this area.
            z_tmp_left = steepness_left[i] * x_tmp
            z_tmp_right = steepness_right[i] * x_tmp

            # Check for which (quantized) x-values the values in the background-lookup-function are greater than the values of the occlusion lines.
            cmp_left = np.sign(old_lookup - z_tmp_left)
            cmp_right = np.sign(old_lookup - z_tmp_right)

            ######Differentiate between 6 cases:
            # case 1: no intersection between z_temp_left and lookup and z_tmp_left and lookup: both steepnesses are negative
            # case 2: no intersection between lookup and z_tmp_left (negative steepness of steepness_min) but intersection between lookup and z_tmp_right
            # case 3: z_tmp_left and z_tmp_right have intersection with lookup --> most common case.
            # case 4: z_tmp_left has intersection, z_tmp_right has no intersection and positive steepness
            # case 5: no intersection of lookup with either z_tmp_left and z_tmp_right, both steepnesses are positive
            # case 6: no intersection of lookup with either z_tmp_left and z_tmp_right, both steepness_max > 0 and steepness_left < 0
            idx_intersection_left = 0
            idx_intersection_right = 0

            if cmp_right.max != cmp_right.min:
                # cases 2 and 3

                # idx_intersection_right = np.argmax(-cmp_right[0] * cmp_right)
                if cmp_right[-1] < 0:
                    idx_intersection_right = - np.argmax(np.flipud(cmp_right))
                else:
                    idx_intersection_right = - np.argmin(np.flipud(cmp_right))

                if cmp_left.max != cmp_left.min:
                    # case 3
                    # check for both cmp_tmp_left and cmp_tmp_right where the sign is switched
                    idx_intersection_left = np.argmax(-cmp_left[0] * cmp_left)
                else:
                    # case 2, no need to adapt idx_intersection_left.
                    pass

            else:
                # cases 1, 4, 5, 6
                if steepness_right < 0:
                    # case 1: there is no occlusion of the background points here, do not set entries in the block_vector to True....
                    pass
                else:
                    # cases 4, 5, 6
                    idx_intersection_right = cmp_right.shape[0]

                    if cmp_left.max != cmp_left.min:
                        # case 4, check for a sign switch in the vector cmp_tmp_left
                        idx_intersection_left = np.argmax(-cmp_left[0] * cmp_left)
                    else:
                        # cases 5,6
                        if steepness_left < 0:
                            # case 6, no need to adapt idx_intersection_left.
                            pass
                        else:
                            # case 5 --> idx_intersection_left = length of cmp_left.
                            idx_intersection_left = cmp_left.shape[0]

            #######################################################################################
            ## End of 6 cases   ###################################################################

            # Mark the block_vector as True where possible occlusion from a cluster in the foreground occurs and influences the points in the background.
            block_vector[idx_intersection_left:idx_intersection_right] = True

        return block_vector

    def adapt_lookup_functions_for_comparison(self, old_x_min, old_x_max, old_lookup, new_x_min, new_x_max, new_lookup):
        #Adapt the two lookup-tables in a manner that they begin and end at the same indices. Work with concatenations.

        start_index_old = 0
        end_index_old = old_lookup.shape[0]
        start_index_new = 0
        end_index_new = new_lookup.shape[0]

        #Initialize concat-vectors to None. If not none, a concatenation is carried out later.
        concat_left = None
        concat_right = None

        # Check for the left side.
        if new_x_min == old_x_min:
            pass
        #If new_x_min > old_x_min: concatenate the n first values from the old_lookup table to the new_lookup table and change the start index accordingly.
        elif new_x_min > old_x_min:
            n = int((new_x_min - old_x_min) / self.step_size)
            concat_left = old_lookup[0:n]
            new_x_min = old_x_min
            start_index_old = n #When compoaring both lookup-tables, the start index for the old lookup table will start at value n.
        else:
            # Otherwise: concatenate the n first values from the new_lookup table to the old_lookup table and change the start index accordingly.
            n = int((old_x_min - new_x_min) / self.step_size)
            concat_left = new_lookup[0:n]
            start_index_new = n

        # Check for the right side.
        if new_x_max == old_x_max:
            pass
        elif new_x_max > old_x_max:
            n = int((new_x_max - old_x_max) / self.step_size)
            concat_right = new_lookup[-n:]
            end_index_new = -n #When compoaring both lookup-tables, the end index for the new lookup table will be shifted to the left by n indices.
        else:
            n = int((old_x_max - new_x_max) / self.step_size)
            concat_right = old_lookup[-n:]
            new_x_max = old_x_max
            end_index_old = -n

        #return the new indices, min and max values and also the concatenations for the left and the right side of the lookup-table.
        return start_index_new, start_index_old, end_index_new, end_index_old, new_x_min, new_x_max, concat_left, concat_right

    def compare_bg_functions(self, point_cloud, old_dict, new_dict, use_clusters=True, clusters = {}, mode ='forward'):

        #Compare two background functions
        old_x_min, old_x_max, old_lookup = old_dict['x_min'], old_dict['x_max'], old_dict['lookup_table']
        new_x_min, new_x_max, new_lookup = new_dict['x_min'], new_dict['x_max'], new_dict['lookup_table']

        #If use_clusters=True: Take into account occlusions which stem from clusters in the foreground and ignore this specific
        #part of the background function (i.e.: do not adapt the part of the background function which is occluded).
        if use_clusters == True:
            block_vector = self.calculate_block_vector_for_occlusion(old_lookup, old_x_min, old_x_max, clusters)
        else:
            block_vector = np.zeros_like(old_lookup, dtype=bool)

        #Calculate crucial indices which are needed for comparing the two lookup tables with each other.
        start_index_new, start_index_old, end_index_new, end_index_old, new_x_min, new_x_max, concat_left, concat_right = \
            self.adapt_lookup_functions_for_comparison(old_x_min, old_x_max, old_lookup, new_x_min, new_x_max, new_lookup)

        block_vector = block_vector[start_index_old:end_index_old]  # adapt thye block_vector to point to the correct x-values.
        tmp_lookup_new = new_dict['lookup_table'][start_index_new:end_index_new]
        tmp_lookup_new[block_vector] = 0  # Set to 0, so that the background fucntion will not be changed at these points.

        if mode == 'backward':
            # Background function should "move back", when the closest point for an x-value is further away (bigger z-value)
            # than the current z-value of the current background-function. This allows an adaptive background function
            # in which points which move (e.g. persons, etc.) will become part of "foreground" after some iterations.
            cmp_lookup = np.max(np.vstack((old_lookup[start_index_old:end_index_old], tmp_lookup_new)), axis = 0)
        else:
            #Forward mode. Keep the point closest to the camera (smallest z-value in that case) (in contrast to backward mode,
            #where the furthers point is kept). Useful when dealing with noise from the depth camera.
            cmp_lookup = np.min(np.vstack((old_lookup[start_index_old:end_index_old], tmp_lookup_new)), axis=0)

        #Values which have been discared for the sake of comparison are added again.
        if concat_left is not None:
            cmp_lookup = np.concatenate((concat_left,cmp_lookup)) #Concatenate on the left side of the lookup table.

        if concat_right is not None:
            cmp_lookup = np.concatenate((cmp_lookup, concat_right))

        return new_x_min, new_x_max, cmp_lookup



    def finetune_bg_function_for_noise(self, point_cloud):

        #old_dict is background function currently saved in the tmp_background dict variable.
        #Calculate the current background parameters and put them to new_dict.

        old_dict = {}
        old_dict['x_min'], old_dict['x_max'], old_dict['lookup_table'] = self.deepcopy_params()
        self.init_bg_removal_function(point_cloud)

        #Compare old_dict to new_dict and keep the z-values closest to the camera when comparing both of them. This gives a more
        #robust background function which is less sensitive to noise.
        new_x_min, new_x_max, cmp_lookup  = \
            self.compare_bg_functions(point_cloud, old_dict=old_dict, new_dict=self.tmp_background, use_clusters=False, clusters={}, mode='forward')

        #Assign new values to the tmp_background dict.
        self.tmp_background['x_min'] = new_x_min
        self.tmp_background['x_max'] = new_x_max
        self.tmp_background['lookup_table'] = cmp_lookup

    def finetune_bg_function_for_moving_objects(self, point_cloud, clusters):

        #Compare old_dict to new_dict and keep the z-values furhert away from the camera. This is used in order to adapt the background
        #functions for moving objects, meaning that moving objects, which have wrongly assumed as background, are removed from the background.
        new_x_min, new_x_max, cmp_lookup = \
            self.compare_bg_functions(point_cloud, old_dict=self.background_dict, new_dict=self.tmp_background, use_clusters=True, clusters=clusters,
                                      mode='backward')
        self.tmp_background['x_min'] = new_x_min
        self.tmp_background['x_max'] = new_x_max
        self.tmp_background['lookup_table'] = cmp_lookup