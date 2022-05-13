import copy
import time
import numpy as np

from pyeyeengine.background.background import BackgroundFunction
from pyeyeengine.point_cloud.point_cloud import PointCloud
from pyeyeengine.center_of_mass.point_cluster import PointCluster
from pyeyeengine.center_of_mass.center_of_mass import CenterOfMass
from pyeyeengine.utilities.astra_orbbec_conf import AstraOrbbec
from pyeyeengine.center_of_mass.floor_extractor import FloorExtractor
from pyeyeengine.center_of_mass.jump_detector import JumpDetector
import pyeyeengine.utilities.global_params as Globals
from pyeyeengine.utilities.logging import Log
from pyeyeengine.camera_utils.frame_manager import FrameManager

#If an accelerometer is present, use an accelerometer, otherwise set usage to FALSE.
try:
    from pyeyeengine.accelerometer.accelerometer_functions import get_total_angle_obie
    USE_ACCELEROMETER = True
except:
    USE_ACCELEROMETER = False


class COMEngine:
    def __init__(self):
        self.engine_type = Globals.EngineType.COM
        self._camera_conf = AstraOrbbec(scale_factor = 2)
        if USE_ACCELEROMETER == True:
            self.x_angle = get_total_angle_obie(iterations=50)
        else:
            # self.x_angle = 67.5
            self.x_angle = 0

        FrameManager.getInstance().set_depth_resolution(Globals.Resolution((self._camera_conf.y_res, self._camera_conf.x_res)))
        depth_image = FrameManager.getInstance().get_depth_frame()
        self.pcl = PointCloud(depth_image, x_rotation=self.x_angle, y_rotation=0, camera_conf=self._camera_conf)
        self.floor_extractor = FloorExtractor(self.pcl.point_cloud, alg_conf={'eps': 150, 'min_samples': 100}, percentage=0.2)
        #Margin which defines the minimum distance which a euclidean point needs to have from the floor plane in order to be kept as valid.
        self.three_d_y_margin = 150
        if self.x_angle != 0:
            self.pcl.filter_by_function_3D(coord=self.floor_extractor.floor_normal_coord, margin=self.three_d_y_margin)
        # self.alg_conf = {'eps': 150, 'min_samples': 700}
        self.max_z_value = np.max(self.pcl.point_cloud[:, 2])
        self.bg_function = BackgroundFunction(self.pcl.point_cloud, step_size= 50, padding = 5)
        self.cls = PointCluster(quantize_step_size=100, quant_thr = 25, min_value_cluster = 70, min_size_cluster = 4)
        self.com = CenterOfMass(x_rotation=-self.x_angle ,y_rotation=0)
        if self.x_angle != 0:
            # self.jumps = JumpDetector(y_diff_grad_0=100, y_max_grad_1=45, timeframe=25)
            self.jumps = JumpDetector()
        else:
            # self.jumps = JumpDetector(y_diff_grad_0=100, y_max_grad_1=25, timeframe=25)
            self.jumps = JumpDetector()

        # Margin which defines the minimum distance which a euclidean point needs to have from the background function in order to be kept as valid.
        self.bkgr_margin = 100
        self.bg_finetune_iter_start = 2
        self.bg_finetune_iter_run = 3

        self.bkgr_update_cntr = -1
        self.bkgr_update_thr = 25

    def get_new_point_cloud(self):
        # Get depth image and calculate 3D-points in euclidean space from it
        FrameManager.getInstance().set_depth_resolution(
            Globals.Resolution((self._camera_conf.y_res, self._camera_conf.x_res)))
        depth_image = FrameManager.getInstance().get_depth_frame()
        self.pcl.update_point_cloud(depth_image)

        # X_Angle != 0 means that camera is in OBIE-mode (camera at ceiling). 3D-points belonging to the floor need to be filtered out in order to not intefere with object clustering.
        if self.x_angle != 0:
            self.pcl.filter_by_function_3D(coord=self.floor_extractor.floor_normal_coord, margin=self.three_d_y_margin)

        return depth_image

    def define_bg(self):
        depth_image = self.get_new_point_cloud()

        #initialize background function
        self.bg_function.init_bg_removal_function(self.pcl.point_cloud)

        #Finetune the background function based on several depth_image takes in order to come up for noise wbhich can occur in a single frame.
        for i in range(self.bg_finetune_iter_start):
            FrameManager.getInstance().set_depth_resolution(
                Globals.Resolution((self._camera_conf.y_res, self._camera_conf.x_res)))
            depth_image = FrameManager.getInstance().get_depth_frame()
            self.pcl.update_point_cloud(depth_image)
            self.bg_function.finetune_bg_function_for_noise(self.pcl.point_cloud)

        #Adopt the new parameters into the background function.
        self.bg_function.update_background_function()

    def get_com(self):
        depth_image = self.get_new_point_cloud()

        #points_unfiltered contain the 3D-point cloud before background removal (but after floor filtering). They are "put aside" for updating the background.
        if self.bkgr_update_cntr >= self.bkgr_update_thr - self.bg_finetune_iter_run:
            points_unfiltered = copy.deepcopy(self.pcl.point_cloud)

        #Remove background from 3D-point cloud
        self.pcl.filter_by_lookup(self.bg_function.background_dict['lookup_table'], x_min=self.bg_function.background_dict['x_min'],
                             x_max=self.bg_function.background_dict['x_max'], step_size=self.bg_function.step_size, margin=self.bkgr_margin)

        #Perform object cluistering.
        self.cls.apply_clustering(self.pcl.point_cloud, x_max_value=self.bg_function.background_dict['x_max'],
                             x_min_value=self.bg_function.background_dict['x_min'], max_z_value=self.max_z_value)

        #Calculate Centers of mass.
        self.com.calculate_center_of_mass(clusters=self.cls.clusters)

        #Update the background function
        if self.bkgr_update_cntr >= self.bkgr_update_thr - self.bg_finetune_iter_run:

            #At the first iteration, initliaze new background function.
            if self.bkgr_update_cntr == (self.bkgr_update_thr - self.bg_finetune_iter_run):
                self.bg_function.init_bg_removal_function(points_unfiltered)

            #Finetune the background function in the following iterations.
            elif self.bkgr_update_cntr < self.bkgr_update_thr:
                self.bg_function.finetune_bg_function_for_noise(points_unfiltered)

            #Perform a final finetuning step and adopt the new background parameters into the background function --> updated background function is received.
            else:
                self.bg_function.finetune_bg_function_for_noise(points_unfiltered)
                self.bg_function.finetune_bg_function_for_moving_objects(points_unfiltered, clusters=self.cls.clusters)
                self.bg_function.update_background_function()
                bkgr_update_cntr = 0

        self.bkgr_update_cntr += 1

        # Log.d('Centers of mass: {}'.format(self.com.centers_of_mass))
        #Perform a jump detection.
        if self.com.centers_of_mass:

            self.jumps_init, self.jumps_in_game = self.jumps.check_jump(self.com.centers_of_mass)

        #     if len(self.jumps_init) > 0:
        #         for jmp in self.jumps_init:
        #             print('--------------------------------------------------------------------')
        #             print('--------------------------------------------------------------------')
        #             print('OBJECT {} PERFORMED A HALF JUMP'.format(jmp))
        #             print('--------------------------------------------------------------------')
        #             print('--------------------------------------------------------------------')
        #     if len(self.jumps_in_game) > 0:
        #         for jmp in self.jumps_in_game:
        #             print('--------------------------------------------------------------------')
        #             print('--------------------------------------------------------------------')
        #             print('OBJECT {} PERFORMED A FULL JUMP'.format(jmp))
        #             print('--------------------------------------------------------------------')
        #             print('--------------------------------------------------------------------')
        else:
            self.jumps.reset_jump_dict()
            self.jumps_init = []
            self.jumps_in_game = []

        return self.com.centers_of_mass, self.com.com_in_pixels, self.cls, self.jumps_init, self.jumps_in_game