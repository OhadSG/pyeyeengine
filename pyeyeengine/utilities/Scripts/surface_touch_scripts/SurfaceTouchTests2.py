import cv2
import numpy as np
import copy
from skimage import morphology
from skimage.measure import label, regionprops
# from math import atan2, cos, sin, sqrt, pi

from pyeyeengine.utilities.global_params import EngineType
from pyeyeengine.utilities.astra_orbbec_conf import AstraOrbbec
from pyeyeengine.tracking.tracker import TrackedObject
import pyeyeengine.utilities.global_params as GlobalParams
from pyeyeengine.calibration import calibration_utility_functions
from pyeyeengine.eye_engine.key_points_limiter import KeyPointsLimiter
from pyeyeengine.eye_engine.fps_counter import FPSCounter

#Debug imports
# from matplotlib import pyplot as plt
from pyeyeengine.utilities.Scripts.camera_scripts.LocalFrameManager import LocalFrameManager
from pyeyeengine.utilities.Scripts.general_scripts.LocalCalibration import LocalCalibration

class TouchEngine():
    def __init__(self, calibrator, frame_manager, camera_conf = AstraOrbbec(scale_factor=GlobalParams.CAMERA_SCALE_FACTOR)):
        print("Engine Initialized")
        self.engine_type = EngineType.NewTouch
        self._camera_conf = camera_conf
        self._calibrator = calibrator
        self.frame_manager = frame_manager
        self.background_average_factor = 3
        self.background_simple = self.get_simple_background(factor=self.background_average_factor)
        self._key_pts_limiter = KeyPointsLimiter(screen_width=1280, screen_height=800)
        self.binary_final = np.zeros((GlobalParams.DEPTH_MEDIUM_QUALITY.height, GlobalParams.DEPTH_MEDIUM_QUALITY.width), dtype=bool)
        self.fine_search_binary = np.zeros_like(self.binary_final, dtype=bool)
        self._fps_counter = FPSCounter()

        self._background_model = None
        self.background_absolute_cntr = -121
        self.background_total_update_timepoints = [-101, -1, 500]
        self.background_thr = 25
        self.background_avg_nr = 3
        self.background_cntr = -1
        self.background_acc_list = []

        # Tests
        self.diff_map_threshold_far = [0, 255]
        self.diff_map_threshold_near = [1, 1]
        self.minimum_size_objects = 15

        # self.diff_map_threshold_far = [0, 100] #thresholds for finding the back surface of the hand.
        # self.diff_map_threshold_near = [80, 90] #fine thresholds in order to find the hand silhouette.
        # self.minimum_size_objects = 15

        self.window_title = "Result"
        self.slider_max_value = 255
        self.trackbar1_near = "[BIG-NEAR] {}".format(self.diff_map_threshold_far)
        self.trackbar1_far = "[BIG-FAR] {}".format(self.diff_map_threshold_far)
        self.trackbar2_near = "[SMALL-NEAR] {}".format(self.diff_map_threshold_near)
        self.trackbar2_far = "[SMALL-FAR] {}".format(self.diff_map_threshold_near)
        self.trackbar3_size = "[SIZE] {}".format(self.minimum_size_objects)

        cv2.namedWindow(self.window_title)
        cv2.createTrackbar(self.trackbar1_near, self.window_title, self.diff_map_threshold_far[0], self.slider_max_value, self.trackbar1_changed_near)
        cv2.createTrackbar(self.trackbar1_far, self.window_title, self.diff_map_threshold_far[1], self.slider_max_value, self.trackbar1_changed_far)
        cv2.createTrackbar(self.trackbar2_near, self.window_title, self.diff_map_threshold_near[0], self.slider_max_value, self.trackbar2_changed_near)
        cv2.createTrackbar(self.trackbar2_far, self.window_title, self.diff_map_threshold_near[1], self.slider_max_value, self.trackbar2_changed_far)
        cv2.createTrackbar(self.trackbar3_size, self.window_title, self.minimum_size_objects, self.slider_max_value, self.trackbar3_size_changed)

        # self.fig, self.ax = plt.subplots()
        # self.fig2, self.ax2 = plt.subplots()

    def trackbar1_changed_near(self, value):
        self.diff_map_threshold_far[0] = value

    def trackbar1_changed_far(self, value):
        self.diff_map_threshold_far[1] = value

    def trackbar2_changed_near(self, value):
        self.diff_map_threshold_near[0] = value

    def trackbar2_changed_far(self, value):
        self.diff_map_threshold_near[1] = value

    def trackbar3_size_changed(self, value):
        self.minimum_size_objects = value

    def get_simple_background(self, factor = 15):
        #Background: average of #factor background images.
        testlist = []
        background = np.zeros((GlobalParams.DEPTH_MEDIUM_QUALITY.height, GlobalParams.DEPTH_MEDIUM_QUALITY.width))

        # for i in range(10):
        #     _, _, depth_image_masked = self.get_depth_image_and_mask()
        self.get_depth_image_and_mask()

        for i in range(factor):
            _,_, depth_image_masked = self.get_depth_image_and_mask()
            testlist.append(depth_image_masked)
            background += depth_image_masked

        background = background / factor
        return background

    def get_depth_image_and_mask(self):
        depth_image = self.frame_manager.get_depth_frame()
        depth_image = cv2.medianBlur(np.float32(depth_image), 5)
        mask = calibration_utility_functions.get_mask(self._calibrator.warp_mat_cam_2_displayed)/255

        depth_image_masked = depth_image * cv2.dilate(mask, kernel=np.ones((20,20),np.uint8))
        depth_image_masked = np.float32(depth_image * mask)
        depth_image_masked = cv2.medianBlur(depth_image_masked, 5)
        # depth_image_masked = depth_image #DEBUG
        return depth_image, mask, depth_image_masked

    def contour_to_voxels(self, contour):
        # average = [sum(x) / len(x) for x in zip(*contour)]
        # x, y = int(average[0][0]), int(average[0][1])
        x, y = contour[:, 0, 0], contour[:, 0, 1]
        return np.int32(np.vstack((x, y, np.ones_like(x))).T)

    def display_on_image(self, img, binary):
        contours, _ = cv2.findContours(np.uint8(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #version when running on local PC
        # _, contours, _ = cv2.findContours(np.uint8(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        voxels = [self.contour_to_voxels(contour) for contour in contours if contour.shape[0] > 2]

        if len(voxels) > 0:
            # cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
            for key_pts in voxels:
                for key_pt_num in range(key_pts.shape[0]):
                    print("Ci")
                    img = cv2.circle(img, (np.int16(key_pts[key_pt_num, 0]),
                                           np.int16(key_pts[key_pt_num, 1])), 1, (0, 0, 255), -1)

        return img, voxels

    def small_background_update(self, depth_image_masked, thr = 20):
        #Correct points which have been occluded by touch interaction in the big background update.
        if self.background_cntr > self.background_thr:
            self.background_acc_list.append(depth_image_masked)
            arr = np.mean(np.asarray(self.background_acc_list), axis=0)
            diff = arr - self.background_simple

            if np.max(diff) > thr:
                idx = diff > thr
                self.background_simple[idx] = arr[idx]

            self.background_cntr = 0
            self.background_acc_list = []

        elif self.background_cntr + self.background_avg_nr > self.background_thr:
            self.background_acc_list.append(depth_image_masked)

        else:
            pass

    def big_background_update(self):
        #Try to replace the whole background model, except for the points where there is current interaction.
        background_tmp = self.get_simple_background(factor=self.background_average_factor)
        background_diff = np.abs(background_tmp - self.background_simple)
        idx = np.where((background_diff < 100) * (self.fine_search_binary == False), True, False)
        self.background_simple[idx] = background_tmp[idx]
        if self.background_absolute_cntr > 0:
            self.background_absolute_cntr = 0

    def get_fine_search_binary_map(self, binary_map_far):
        #get binary map where to search for hand silhouette.
        # labels_binary_map_far= label(binary_map_far, connectivity=2) #version when running on local PC
        labels_binary_map_far = label(binary_map_far, neighbors=8) #find connected components for binary_map_far
        props = regionprops(labels_binary_map_far)

        self.fine_search_binary = np.zeros_like(self.binary_final, dtype=bool)
        for i in range(len(props)):
            y_diff = props[i].bbox[2] - props[i].bbox[0]
            x_diff = props[i].bbox[3] - props[i].bbox[1]
            factor = 0.6
            y_range= [max(0, int(props[i].bbox[0] - factor * y_diff)), min(int(props[i].bbox[2] + factor*y_diff), GlobalParams.DEPTH_MEDIUM_QUALITY.height)]
            x_range = [max(0, int(props[i].bbox[1] - factor * x_diff)),min(int(props[i].bbox[3] + factor*x_diff), GlobalParams.DEPTH_MEDIUM_QUALITY.width)]
            self.fine_search_binary[y_range[0]:y_range[1], x_range[0]:x_range[1]] = True

    def get_binary_map_far(self, diff_map):
        #Get appproximate area where to search for hand silhouette. In order to deal with noise coming from the camera depth image.
        binary_map_far = cv2.inRange(diff_map, self.diff_map_threshold_far[0], self.diff_map_threshold_far[1])
        binary_map_far = cv2.medianBlur(np.float32(binary_map_far), 5)
        # kernel = np.ones((3, 3), np.uint8)
        # test2 = cv2.morphologyEx(test2, cv2.MORPH_OPEN, kernel)

        binary_map_far = morphology.remove_small_objects(binary_map_far != 0, min_size=20, connectivity=2) #Small objects are removed --> noise.

        return binary_map_far

    def get_binary_final(self, diff_map, mask):
        self.binary_final = cv2.inRange(diff_map, self.diff_map_threshold_near[0], self.diff_map_threshold_near[1]) * self.fine_search_binary * mask
        self.binary_final = cv2.medianBlur(np.float32(self.binary_final), 3)
        # kernel = np.ones((3, 3), np.uint8)
        # self.binary_final = cv2.morphologyEx(np.uint8(self.binary_final), cv2.MORPH_OPEN, kernel)
        self.binary_final = morphology.remove_small_objects(self.binary_final != 0, min_size=15, connectivity=1)

    def get_diff_map(self, depth_image):
        diff_map = self.background_simple - depth_image
        diff_map = cv2.medianBlur(np.float32(diff_map), 5)

        return diff_map


    def process_frame(self):

        self.background_cntr += 1
        self.background_absolute_cntr += 1

        # print('absolute background counter: ', self.background_absolute_cntr)

        if (self.background_absolute_cntr in self.background_total_update_timepoints):
            # print('new background model')
            self.big_background_update()

        depth_image, mask, depth_image_masked = self.get_depth_image_and_mask()

        self.small_background_update(depth_image, thr=self.diff_map_threshold_near[0])

        #Diff_map + morphology
        diff_map = self.get_diff_map(depth_image)

        #binary_map_far: recognize approximate location of hand
        binary_map_far = self.get_binary_map_far(diff_map)

        #Area where to search for hand silhouette
        self.get_fine_search_binary_map(binary_map_far)

        #Final binary map of hand silhouette
        self.get_binary_final(diff_map, mask)

        #Wanted to include orientation of binary map too, but discarded that idea later on.
        # contours, _ = cv2.findContours(np.uint8(test2), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # if len(contours) > 0:
        #     for c in contours:
        #         angle = self.getOrientation(c, test2)

        #Display image with binary touch segmentation map.
        ################################################################################################################
        img = self.frame_manager.get_rgb_frame()
        # self.ax2.cla()
        # self.ax2.imshow(img)
        # self.ax2.imshow(self.binary_final, alpha=0.5)
        # self.fig2.show()
        # plt.pause(0.001)
        ################################################################################################################


        # Display image with points
        ################################################################################################################
        # img_with_points2, voxels = self.display_on_image(copy.deepcopy(img), self.binary_final)
        # self.ax2.cla()
        # self.ax2.imshow(img_with_points2)
        # self.ax2.set_title('new version')
        # self.fig2.show()
        # plt.pause(0.001)
        ################################################################################################################

        #Contours for getting only the outer hand contours.
        contours, _ = cv2.findContours(np.uint8(self.binary_final), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #version when running on local PC
        # _, contours, _ = cv2.findContours(np.uint8(self.binary_final), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        voxels = [self.contour_to_voxels(contour) for contour in contours if contour.shape[0] > 2] #legacy to be aligned with previous code

        # print(voxels)
        # print('min: ', np.round(np.max(diff_map * mask)))

        #legacy code to be aligned with current get_key_points version
        transformed_objects = [TrackedObject(idx, self._key_pts_limiter.apply_limitations(
            self._calibrator.transfrom_points_cam_to_display(key_pts[:, :2])))
                               for idx, key_pts in enumerate(voxels)]

        if len(voxels) > 0:
            # cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
            for key_pts in voxels:
                for key_pt_num in range(key_pts.shape[0]):
                    img = cv2.circle(img, (np.int16(key_pts[key_pt_num, 0]),
                                           np.int16(key_pts[key_pt_num, 1])), 1, (0, 255, 0), -1)

        show_image(self.window_title, img)

        self._fps_counter.process_frame() #legacy code

        return transformed_objects

def show_image(name, image):
    resized = cv2.resize(image, (1280, 800))
    cv2.imshow(name, resized)
    cv2.waitKey(1)

if __name__ == '__main__':
    frame_manager = LocalFrameManager(rgb_resolution=GlobalParams.RGB_HIGH_QUALITY)
    calibrator = LocalCalibration()
    result = calibrator.calibrate(frame_manager)
    print("Calibrated: {}".format(result))
    frame_manager = None
    frame_manager = LocalFrameManager(rgb_resolution=GlobalParams.RGB_LOW_QUALITY, depth_resolution=GlobalParams.DEPTH_MEDIUM_QUALITY)
    engine = TouchEngine(frame_manager=frame_manager, calibrator=calibrator)

    while True:
        engine.process_frame()