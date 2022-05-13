import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from pyeyeengine.camera_utils.camera_manager import CameraManager
from pyeyeengine.utilities.astra_orbbec_conf import AstraOrbbec
from pyeyeengine.point_cloud.point_cloud import PointCloud
from pyeyeengine.background.background import BackgroundFunction
# from pyeyeengine.center_of_mass.point_cluster import PointCluster
from pyeyeengine.center_of_mass.point_cluster import PointCluster
from pyeyeengine.center_of_mass.center_of_mass import CenterOfMass
try:
    from pyeyeengine.accelerometer.accelerometer_functions import get_total_angle_obie
    from pyeyeengine.accelerometer.grove_3_axis_digital_accelerometer import ADXL345
    USE_ACCELEROMETER = True
except:
    USE_ACCELEROMETER = False

from pyeyeengine.utilities.helper_functions import get_best_plane_from_points
from pyeyeengine.center_of_mass.floor_extractor import FloorExtractor
from pyeyeengine.center_of_mass.jump_detector import JumpDetector

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import time





def find_occlusion_vector(cls):

    steepness_max = []
    steepness_min = []

    for keys in cls.clusters:
        points = cls.clusters[keys]['cluster_points']
        max_coord = np.argmax(points[:,0], axis=0)
        min_coord = np.argmin(points[:,0], axis=0)

        steepness_max.append(points[max_coord,2] / points[max_coord,0])
        steepness_min.append(points[min_coord,2] / points[min_coord,0])

    return steepness_max, steepness_min

########################################################################################################################
######################################ACCELEROMETER#####################################################################
obie_angle = 0

if USE_ACCELEROMETER:
    adxl345 = ADXL345()
    angle_list = []
    for i in range(50):
        angle, _, _ = adxl345.getAngles()
        angle_list.append(angle)

    obie_angle = sum(angle_list)/len(angle_list)
    print(obie_angle)
    camera_angle = 67.5
    total_angle = camera_angle - obie_angle

else:
    total_angle = 0

########################################################################################################################
########################################################################################################################
x_rotation = total_angle
y_rotation = 0


cam = CameraManager()
# alg_conf = {'eps' : 150, 'min_samples' : 400}
step_size = 50


camera_conf = AstraOrbbec(scale_factor = 2) #scale factor to maximum resolution of 480 x 640.
depth_image = cam.get_depth(res_xy=(camera_conf.y_res, camera_conf.x_res))


pcl = PointCloud(depth_image, x_rotation=x_rotation, y_rotation=y_rotation, camera_conf=camera_conf)

percentage = 0.2
alg_conf = {'eps' : 150, 'min_samples' : 100}
floor_extractor = FloorExtractor(pcl.point_cloud, alg_conf, percentage=percentage)

# ########################################################################################################################
# # Get lowest points in y-direction and define them as floor
#
# percentage = 0.2
# points = copy.deepcopy(pcl.point_cloud)
# points = points[points[:, 1].argsort(), :]
#
# nr = int(np.round(percentage * points.shape[0]))
# low_points = points[:int(nr)]
#
# low_points_flat = np.transpose(np.array([low_points[:,1], low_points[:,2]])) #oppress the x-direction
#
# ########################################################################################################################
# #perform dbscan for outlier removal
# alg_conf = {'eps' : 150, 'min_samples' : 100}
# dbscan = DBSCAN(eps=alg_conf['eps'], min_samples=alg_conf['min_samples'])
#
# dbscan.fit(low_points_flat)
#
# # Get the biggest cluster from the dbscan algorithm, assume as floor
#
# labels = dbscan.labels_
# unique_labels = set(labels)
#
# if -1 in unique_labels:
#     unique_labels.remove(-1)  # remove the noisy cluster
#
# cluster_sizes = []
# cluster_points = []
#
# #low_points_filtered contain the points which are assumed to belong to the floor.
#
# if len(unique_labels) > 0:
#     for l in unique_labels:
#         class_member_mask = (labels == l)
#         cluster_points.append(low_points[class_member_mask, :])
#         cluster_sizes.append(np.sum(class_member_mask))
#
#     print('Cluster sizes :', cluster_sizes)
#
#     if len(unique_labels) > 1:
#         new_order = np.array(cluster_sizes).argsort()
#         low_points_filtered = cluster_points[new_order[-1]]
#     else:
#         low_points_filtered = cluster_points[0]
# else:
#     print('No clusters found')
#
# #######################################################################################################################
# #######################################################################################################################
#
# #calculate floor normal and distance
# normal, distance, _ = get_best_plane_from_points(low_points_filtered)
# coord = normal
# coord.append(distance)

three_d_y_margin = 150

# Filter the floor out of the function
if x_rotation != 0:
    pcl.filter_by_function_3D(coord=floor_extractor.floor_normal_coord, margin=three_d_y_margin)


#######################################################################################################################

bg_function = BackgroundFunction(pcl.point_cloud, step_size= step_size, padding = 5)
bg_finetune_iter = 3

for i in range(bg_finetune_iter):
    depth_image = cam.get_depth(res_xy=(camera_conf.y_res, camera_conf.x_res))
    pcl.update_point_cloud(depth_image=depth_image)
    if x_rotation != 0:
        pcl.filter_by_function_3D(coord=floor_extractor.floor_normal_coord, margin=three_d_y_margin)
    bg_function.finetune_bg_function_for_noise(pcl.point_cloud)

bg_function.update_background_function()

cls = PointCluster(quantize_step_size=100, quant_thr = 30, min_value_cluster = 75, min_size_cluster = 4)
com = CenterOfMass(x_rotation=-x_rotation ,y_rotation=-y_rotation)

margin = 200

cntr = 0

# fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
#
#begin scatter plot
# fig2,ax2 = plt.subplots()
# x_vals = np.arange(bg_function.background_dict['x_min'], bg_function.background_dict['x_max'] + bg_function.step_size, bg_function.step_size)
# z_vals = bg_function.background_dict['lookup_table'] - margin
#
# ax2.plot(x_vals, z_vals, 'r')
# ax2.scatter(pcl.point_cloud[:, 0], pcl.point_cloud[:, 2], c='b')
# fig2.show()
# #end scatter plot

print('here')

flag = 0
bkgr_update_cntr = -1
bkgr_update_thr = 25
max_z_value = np.max(pcl.point_cloud[:,2])

x_list = []
y_list = []

if x_rotation != 0:
    jumps = JumpDetector(y_diff_grad_0=100, y_max_grad_1 = 45, timeframe=40)
else:
    jumps = JumpDetector(y_diff_grad_0=100, y_max_grad_1=25, timeframe=40)

while (True):
# for i in range(300):

    start = time.time()
    #scatter plot
    # ax2.cla()
    # ax2.plot(x_vals, z_vals, 'r')
    #
    # ax.cla()
    # ax.set_ylim([0, 240])
    # ax.set_xlim([0, 320])
    #
    #
    rgb_image = cam.get_rgb(res_xy=(camera_conf.y_res, camera_conf.x_res))
    # rotated_rgb = np.rot90(rgb_image, 2)
    # ax.imshow(rgb_image)

    test1 = time.time()
    depth_image = cam.get_depth(res_xy=(camera_conf.y_res, camera_conf.x_res))
    test2 = time.time()
    # print('Time for getting depth image: ', test2 - test1)

    test1 = time.time()
    pcl.update_point_cloud(depth_image)
    test2 = time.time()
    # print('Time for updating point cloud: ', test2 - test1)
    # filter out the floor
    test1 = time.time()
    if x_rotation != 0:
        pcl.filter_by_function_3D(coord=floor_extractor.floor_normal_coord, margin=three_d_y_margin)
    test2 = time.time()
    # print('Time for filtering 3D is: ', test2 - test1)

    if bkgr_update_cntr >= bkgr_update_thr - bg_finetune_iter:
        points_unfiltered = copy.deepcopy(pcl.point_cloud)



    # ax2.scatter(pcl.point_cloud[:, 0], pcl.point_cloud[:, 2], c='b')

    test1 = time.time()
    pcl.filter_by_lookup(bg_function.background_dict['lookup_table'], x_min= bg_function.background_dict['x_min'],
                         x_max=bg_function.background_dict['x_max'], step_size= bg_function.step_size, margin=margin)
    test2 = time.time()
    # print('Time for filtering point cloud: ', test2 - test1)
    # if ((cntr % 1000) == 0 or len(cls.clusters) == 0) :
    #     cls.apply_clustering(point_cloud=pcl.point_cloud)
    #     print('doing DBSCAN')
    # else:
    #     cls.apply_cluster_tracking(point_cloud=pcl.point_cloud, window_margin=0.4)
    #     print('doing TRACKING')
    test1 = time.time()
    cls.apply_clustering(pcl.point_cloud, x_max_value=bg_function.background_dict['x_max'],
                         x_min_value=bg_function.background_dict['x_min'], max_z_value=max_z_value)
    test2 = time.time()
    # print('Time for clustering: ', test2 - test1)

    # print(bkgr_update_cntr)
    test1 = time.time()
    com.calculate_center_of_mass(clusters=cls.clusters)
    test2 = time.time()
    # print('Time for calculating center of mass: ', test2 - test1)

    if bkgr_update_cntr >= bkgr_update_thr - bg_finetune_iter:

        if bkgr_update_cntr == (bkgr_update_thr - bg_finetune_iter):
            bg_function.init_bg_removal_function(points_unfiltered)

        elif bkgr_update_cntr < bkgr_update_thr:
            bg_function.finetune_bg_function_for_noise(points_unfiltered)

        else:
            bg_function.finetune_bg_function_for_noise(points_unfiltered)
            bg_function.finetune_bg_function_for_moving_objects(points_unfiltered, clusters=cls.clusters)
            bg_function.update_background_function()
            bkgr_update_cntr = 0
            x_vals = np.arange(bg_function.background_dict['x_min'], bg_function.background_dict['x_max'] + bg_function.step_size, bg_function.step_size)
            z_vals = bg_function.background_dict['lookup_table'] - margin
            print('updating background')

    # print('Centers of mass:')
    # print(com.centers_of_mass)
    jumps_init, jumps_in_game = jumps.check_jump(com.centers_of_mass)

    if len(jumps_init) > 0:
        for jmp in jumps_init:
            print('--------------------------------------------------------------------')
            print('--------------------------------------------------------------------')
            print('OBJECT {} PERFORMED A HALF JUMP'.format(jmp))
            print('--------------------------------------------------------------------')
            print('--------------------------------------------------------------------')

    if len(jumps_in_game) > 0:
        for jmp in jumps_in_game:
            print('--------------------------------------------------------------------')
            print('--------------------------------------------------------------------')
            print('OBJECT {} PERFORMED A FULL JUMP'.format(jmp))
            print('--------------------------------------------------------------------')
            print('--------------------------------------------------------------------')

        # time.sleep(0.2)



    # time.sleep(0.2)


    # print('And Pixel values:')
    # print(com.com_in_pixels)

    # print('Cluster IDs:')

    # if len(cls.objects_in) > 0 or len(cls.objects_out) >0:
    #     print('Objects in: ', cls.objects_in)
    #     print('Objects out: ', cls.objects_out)

    # for i in range(len(cls.clusters)):
    #     print(cls.clusters.keys())

    bkgr_update_cntr += 1




    # if len(cls.clusters) > 0:
        # print('Number of found clusters are: ', len(cls.clusters))

        # x_list.append(com.centers_of_mass[1][0])
        # y_list.append(com.centers_of_mass[1][1])
        # with open('/home/ron/Documents/testfolder/x_list_flatjump_ron_4.pkl', 'wb') as output:
        #     pickle.dump(x_list, output, pickle.HIGHEST_PROTOCOL)
        # with open('/home/ron/Documents/testfolder/y_list_flatjump_ron_4.pkl', 'wb') as output:
        #     pickle.dump(y_list, output, pickle.HIGHEST_PROTOCOL)
        # with open('/usr/local/lib/python3.5/dist-packages/pyeyeengine/testfolder/x_list_jump_efi_4.pkl', 'wb') as output:
        #     pickle.dump(x_list, output, pickle.HIGHEST_PROTOCOL)
        # with open('/usr/local/lib/python3.5/dist-packages/pyeyeengine/testfolder/y_list_jump_efi_4.pkl', 'wb') as output:
        #     pickle.dump(y_list, output, pickle.HIGHEST_PROTOCOL)
        # with open('/usr/local/lib/python3.5/dist-packages/pyeyeengine/testfolder/rgb_' + str(cntr) +'.pkl', 'wb') as output:
        #     pickle.dump(rgb_image, output, pickle.HIGHEST_PROTOCOL)
        #
        # with open('/usr/local/lib/python3.5/dist-packages/pyeyeengine/testfolder/com_' + str(cntr) +'.pkl', 'wb') as output:
        #     pickle.dump(com, output, pickle.HIGHEST_PROTOCOL)



    # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, 10)]
    #
    # steepness_max, steepness_min = find_occlusion_vector(cls)
    #
    # for m, key in enumerate(cls.clusters):
    #     # color = colors[cls.clusters[m]['cluster_id'] - 1]
    #     color = colors[key - 1]
    #     ax2.scatter(cls.clusters[key]['cluster_points'][:, 0], cls.clusters[key]['cluster_points'][:, 2], c=color)
    #
    #     x_tmp = np.arange(bg_function.background_dict['x_min'], bg_function.background_dict['x_max'], 1)
    #     ax2.plot(x_tmp, x_tmp * steepness_max[m], 'k')
    #     ax2.plot(x_tmp, x_tmp * steepness_min[m], 'k')
    #
    # ax2.set_ylim((0, max_z_value))
    # ax2.set_xlim((bg_function.background_dict['x_min'], bg_function.background_dict['x_max']))
    # plt.savefig('/usr/local/lib/python3.5/dist-packages/pyeyeengine/testfolder/fig_' + str(cntr) +'.png')

    # fig2.show()
    # print(flag)
    # if flag == 1:
    #     circle1.remove()
    #     flag=0
    #
    # if len(com.com_in_pixels)>0:
    #     circle1 = plt.Circle((320 - com.com_in_pixels[0,1], 240 - com.com_in_pixels[0,0]), 5, color='r')
    #
    #     ax.add_artist(circle1)
    #     fig.show()
    #     flag = 1


    plt.pause(0.001)
    cntr += 1
    # print('Counter:    ',cntr)
    end = time.time()
    # print('Loop time in seconds is: ', end - start)

    #
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


print('finish')



