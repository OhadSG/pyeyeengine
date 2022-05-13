import copy
import matplotlib.pyplot as plt
import numpy as np
from pyeyeengine.camera_utils.camera_manager import CameraManager
from pyeyeengine.utilities.astra_orbbec_conf import AstraOrbbec
from pyeyeengine.point_cloud.point_cloud import PointCloud
from pyeyeengine.background.background import BackgroundFunction
# from pyeyeengine.center_of_mass.point_cluster import PointCluster
from pyeyeengine.center_of_mass.point_cluster import PointCluster
from pyeyeengine.center_of_mass.center_of_mass import CenterOfMass

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import time
import pickle





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


cam = CameraManager()
# alg_conf = {'eps' : 150, 'min_samples' : 400}
step_size = 50


camera_conf = AstraOrbbec(scale_factor = 2) #scale factor to maximum resolution of 480 x 640.
depth_image = cam.get_depth(res_xy=(camera_conf.y_res, camera_conf.x_res))


pcl = PointCloud(depth_image, camera_conf=camera_conf)

bg_function = BackgroundFunction(pcl.point_cloud, step_size= step_size, padding = 5)
bg_finetune_iter = 3

for i in range(bg_finetune_iter):
    depth_image = cam.get_depth(res_xy=(camera_conf.y_res, camera_conf.x_res))
    pcl = PointCloud(depth_image, camera_conf=camera_conf)
    bg_function.finetune_bg_function_for_noise(pcl.point_cloud)

bg_function.update_background_function()

cls = PointCluster(quantize_step_size=100, quant_thr = 30, min_value_cluster = 100, min_size_cluster = 4)
com = CenterOfMass()

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
#end scatter plot

print('here')

flag = 0
bkgr_update_cntr = -1
bkgr_update_thr = 25
max_z_value = np.max(pcl.point_cloud[:,2])

x_list = []


while (True):
# for i in range(500):

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
    # rgb_image = cam.get_rgb(res_xy=(camera_conf.y_res, camera_conf.x_res))
    # rotated_rgb = np.rot90(rgb_image, 2)
    # ax.imshow(rotated_rgb)

    test1 = time.time()
    depth_image = cam.get_depth(res_xy=(camera_conf.y_res, camera_conf.x_res))
    test2 = time.time()
    # print('Time for getting depth image: ', test2 - test1)

    test1 = time.time()
    pcl.update_point_cloud(depth_image)
    test2 = time.time()
    # print('Time for updating point cloud: ', test2 - test1)

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
    com.calculate_center_of_mass(clusters=cls.clusters)

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

    print('Centers of mass:')
    print(com.centers_of_mass)
    # print('And Pixel values:')
    # print(com.com_in_pixels)

    # print('Cluster IDs:')
    #
    # if len(cls.objects_in) > 0 or len(cls.objects_out) >0:
    #     print('Objects in: ', cls.objects_in)
    #     print('Objects out: ', cls.objects_out)

    for i in range(len(cls.clusters)):
        print(cls.clusters.keys())

    bkgr_update_cntr += 1




    if len(cls.clusters) > 0:
        print('Number of found clusters are: ', len(cls.clusters))

        x_list.append(com.centers_of_mass[1][0])
        with open('/home/ron/Documents/development/pyeyeengine/testfolder/x_list_4.pkl', 'wb') as output:
            pickle.dump(x_list, output, pickle.HIGHEST_PROTOCOL)

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
    #
    #
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


    # plt.pause(0.001)
    cntr += 1
    print('Counter:    ',cntr)
    end = time.time()
    # print('Loop time in seconds is: ', end - start)

    #
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


print('finish')



