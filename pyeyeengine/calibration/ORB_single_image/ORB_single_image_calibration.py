import cv2
import time
import numpy as np
import os
import sys
import copy
from threading import Thread

from pyeyeengine.point_cloud.point_cloud import PointCloud
from pyeyeengine.utilities import global_params as GlobalParams
from pyeyeengine.utilities import helper_functions as Helper
from pyeyeengine.utilities.logging import Log
from pyeyeengine.utilities.file_uploader import FileUploader
from pyeyeengine.camera_utils.frame_manager import FrameManager
import pyeyeengine.projector_controller.projector_tools as projector_tools
from pyeyeengine.camera_utils.frame_stream.rgb_frame_stream import RgbFrameStream

###### CONSTANTS ######
SAVE_RESULT = True
CALIBRATION_DEBUG = False

REQUIRED_CONFIDENCE = 0.2
MAX_MATCHES = 1000
GOOD_MATCH_PERCENT = 0.2
NORMAL_AVERAGING_ITERATIONS = 5
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
TEMPLATE_IMAGE_PATH = BASE_PATH + ""
DEFAULT_TEMPLATE_IMAGE_PATH = BASE_PATH + "/calibration_template.jpg"
FOUND_MATHCHES_IMAGE_NAME = BASE_PATH + "/calibration_images/matches.jpg"
CALIBRATION_RESULT_IMAGE_NAME = BASE_PATH + "/calibration_images/finished_calibration.jpg"
CROPPED_DEPTH_IMAGE_PATH = BASE_PATH + "/calibration_images/masked_screen.png"
HOMOGRAPHY_FILE_NAME = BASE_PATH + "/../warp_mat_cam_2_displayed.npy"
PLANE_FILE_NAME = BASE_PATH + "/../calibrated_plane.npy"
MASK_FILE_NAME = BASE_PATH + "/../calibrated_mask.npy"
MAX_CALIBRATION_RETRIES = 10
###### CONSTANTS ######

class SingleImageCalibrator:
    def __init__(self, frame_manager: FrameManager, screen_resolution=GlobalParams.Resolution(1280, 800)):
        self.frame_manager = frame_manager
        self.screen_resolution = screen_resolution
        self.warp_mat_cam_2_displayed = None
        self.calibrate_success = False
        self.calibration_scale_factor = GlobalParams.CALIBRATION_SCALE_FACTOR
        self.table_mask = np.uint8(np.ones((960, 1280, 3)) * 255)

        # Plane Normal
        # self.normals_array = []
        # self.plane_normal = [0.0, 0.0, 0.0]
        # self.plane_distance = 0.0
        # self.mask = None

        self.load_saved_calibration()
        self.prepare_templates()

    def load_saved_calibration(self):
        if os.path.isfile(HOMOGRAPHY_FILE_NAME):
            Log.i("[CALIBRATION] Loading saved calibration")
            self.warp_mat_cam_2_displayed = np.load(HOMOGRAPHY_FILE_NAME)
        else:
            Log.w("[CALIBRATION] Engine is not calibrated, please run calibration!")
            self.warp_mat_cam_2_displayed = np.eye(3)

        # if os.path.isfile(PLANE_FILE_NAME) and os.path.isfile(MASK_FILE_NAME):
        #     calibrated_plane = np.load(PLANE_FILE_NAME, allow_pickle=True)
        #     self.mask = np.load(MASK_FILE_NAME)
        #
        #     if len(calibrated_plane) != 2:
        #         # # Log.e("Could not load calibrated normal. Please run calibration again")
        #         self.plane_normal = [0.0, 0.0, 0.0]
        #         self.plane_distance = 0.0
        #     else:
        #         self.plane_normal = calibrated_plane[0]
        #         self.plane_distance = calibrated_plane[1]
        # else:
        #     # # Log.e("Engine is not calibrated! Please run calibration!")

    def get_template_image(raw_image=False):
        Log.d("[CALIBRATION] Delivering template image to admin")
        template = cv2.imread(DEFAULT_TEMPLATE_IMAGE_PATH)
        template = SingleImageCalibrator.edit_template(template)

        if raw_image:
            return cv2.cvtColor(template, cv2.COLOR_RGB2BGR)
        else:
            return cv2.imencode(".png", template)[1].tobytes()

    def edit_template(template):
        modded_template = template

        if CALIBRATION_DEBUG:
            modded_template = Helper.frame_image(template, 2)
            cv2.imwrite(BASE_PATH + "/calibration_images/admin_template.png", modded_template)

        return modded_template

    def prepare_templates(self):
        template = cv2.imread(TEMPLATE_IMAGE_PATH)
        self.template_image = template if template is not None else cv2.imread(DEFAULT_TEMPLATE_IMAGE_PATH)

        assert self.template_image is not None, "Could not load template image from {} or {}".format(
            TEMPLATE_IMAGE_PATH, DEFAULT_TEMPLATE_IMAGE_PATH)

        self.matching_template = SingleImageCalibrator.edit_template(self.template_image)
        self.padded_10_template = np.pad(self.matching_template, ((10, 10), (10, 10), (0, 0)), 'constant', constant_values=(0,))
        self.padded_20_template = np.pad(self.matching_template, ((20, 20), (20, 20), (0, 0)), 'constant', constant_values=(0,))

    def finalize(self):
        # self.find_plane_normal()
        self.persist_calibration()

    def persist_calibration(self):
        Log.d("[CALIBRATION] Saving Calibration to File")
        # calibrated_plane = [self.plane_normal, self.plane_distance]
        # np.save(PLANE_FILE_NAME, calibrated_plane)
        # np.save(MASK_FILE_NAME, self.mask)
        np.save(HOMOGRAPHY_FILE_NAME, self.warp_mat_cam_2_displayed)
        Log.i("[CALIBRATION] Done.")

    def calibrate_with_screen_setter(self, screen_setter=None, table_mask=None, mode="table", calibration_scale_factor = GlobalParams.CALIBRATION_SCALE_FACTOR , sync_normal_calculation = True, delay=0):
        template_image = SingleImageCalibrator.get_template_image(raw_image=True)

        # if table_mask is not None:
        #     template_image = cv2.bitwise_and(template_image, template_image, mask=table_mask)

        screen_setter.set_image(template_image)
        time.sleep(delay)
        return self.calibrate(mode, calibration_scale_factor, sync_normal_calculation, table_mask)

    def calibrate(
            self,
            mode="table",
            calibration_scale_factor = GlobalParams.CALIBRATION_SCALE_FACTOR,
            sync_normal_calculation = True,
            table_mask=np.uint8(np.ones((960, 1280, 3)) * 255)
    ):
        if projector_tools.supports_function(projector_tools.FUNCTION_FOCUS):
            did_succeed = projector_tools.auto_focus()
            if not did_succeed:
                self.calibrate_success = False
                Log.e("[CALIBRATION] New Calibration Failed", extra_details={"reason": "auto-focus failed"})
                return

        Log.i("[CALIBRATION] Starting New Calibration", extra_details={"mode":mode,"resolution":"{}x{}".format(self.screen_resolution.width, self.screen_resolution.height)})

        template_image = self.matching_template

        # if table_mask is not None:
        #     cv2.imwrite(BASE_PATH + "/calibration_images/calibration_table_mask.png", table_mask)
        #     #check mask size!
        #     template_image = cv2.bitwise_and(template_image, template_image, mask=table_mask)

        self.table_mask = table_mask

        # if FrameManager.getInstance().is_projector_on() == False:
        #     Log.e("The projector is off, cannot continue with calibration")
        #     self.calibrate_success = False
        #     return
        # else:
        #     Log.d("The projector is on, continuing with calibration")

        self.frame_manager.rgb_stream.set_resolution_named('large')

        retry_attempt = 0
        self.calibration_scale_factor = calibration_scale_factor

        while True:
            retry_attempt = retry_attempt + 1

            camera_view = self.frame_manager.rgb_stream.get_frame()

            cv2.imwrite(BASE_PATH + "/calibration_images/reference_rgb_frame.png", camera_view)
            cv2.imwrite(BASE_PATH + "/calibration_images/comparison_template.png", template_image)

            template_confidence, homography, warped = self.find_homography(camera_view, [template_image])

            if template_confidence > REQUIRED_CONFIDENCE:
                if CALIBRATION_DEBUG:
                    Log.d("Final Homography:\n{}".format(homography))

                np.save(HOMOGRAPHY_FILE_NAME, homography)
                self.warp_mat_cam_2_displayed = homography

                if sync_normal_calculation:
                    self.finalize()
                else:
                    thread = Thread(target=self.finalize)
                    thread.start()

                FileUploader.save_image(camera_view, "good_calibration.png", "calibration/")

                Log.i("[CALIBRATION] New Calibration Successful", extra_details={"confidence":"{:.1f}".format(template_confidence * 100)})

                self.calibrate_success = True

                return
            elif retry_attempt >= MAX_CALIBRATION_RETRIES:
                FileUploader.save_image(camera_view, "bad_calibration.png", "calibration/")

                Log.e("[CALIBRATION] New Calibration Failed", extra_details={"confidence":"{:.1f}".format(template_confidence * 100)})

                self.calibrate_success = False

                return

    def compare_projection_with_template(self, projected, template, index, factor=1):
        projected_gray = cv2.cvtColor(projected, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        orbBF = cv2.ORB_create(MAX_MATCHES)
        kp1, des1 = orbBF.detectAndCompute(projected_gray, None)
        kp2, des2 = orbBF.detectAndCompute(template_gray, None)

        bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, True)
        matchesBF = bfMatcher.match(des1, des2)
        matchesBF = sorted(matchesBF, key=lambda x: x.distance)
        numGoodMatches = int(len(matchesBF) * GOOD_MATCH_PERCENT)
        matchesBF = matchesBF[:numGoodMatches]

        points1 = np.zeros((len(matchesBF), 2), dtype=np.float32)
        points2 = np.zeros((len(matchesBF), 2), dtype=np.float32)

        for i, match in enumerate(matchesBF):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        try:
            homography, status = cv2.findHomography(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=7, maxIters=10000, confidence=0.99)
        except:
            return 0.0, None, None

        if CALIBRATION_DEBUG:
            print("Homography:\n{}".format(homography))

        height, width, channels = template.shape
        warped_image = cv2.warpPerspective(projected, homography, (width, height))

        template_confidence = cv2.matchTemplate(template, warped_image, method=cv2.TM_CCOEFF_NORMED)
        ransac_confidence = np.sum(status) / len(matchesBF)

        points1 /= factor
        homography, status = cv2.findHomography(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=7,
                                                maxIters=10000, confidence=0.99)

        if SAVE_RESULT:
            matches_image = cv2.drawMatches(projected, kp1, template, kp2, matchesBF, None, flags=2)
            cv2.imwrite(FOUND_MATHCHES_IMAGE_NAME, matches_image)
            pheight, pwidth, _ = projected.shape
            projected_resized = cv2.resize(projected, (int(pwidth / factor), int(pheight / factor)))
            result_image = cv2.warpPerspective(projected_resized, homography, (width, height))
            new_path = "{}/calibration_images/homography_#{}.jpg".format(BASE_PATH, index)
            cv2.imwrite(new_path, result_image)
            print("Homography Confidence: [RANSAC: {} Template: {}]".format(ransac_confidence * 100, template_confidence * 100))

        return template_confidence[0][0], warped_image, homography

    def find_homography(self, cam_image, templates):
        assert len(templates) > 0, "Finding homography requires at least one template"

        Log.d("[CALIBRATION] Finding homography using {} template(s)...".format(len(templates)))

        template_confidence, warped_image, homography = ([], cam_image, [])

        if len(templates) == 1:
            template_confidence, final_warp, final_homography = self.compare_projection_with_template(cam_image, templates[0], index=1, factor=self.calibration_scale_factor)
        else:
            for index, template in enumerate(templates):
                template_confidence, warped_image, homography = self.compare_projection_with_template(warped_image, template, index)

            _, final_warp, final_homography = self.compare_projection_with_template(cam_image, warped_image, 9, factor=self.calibration_scale_factor)

        return template_confidence, final_homography, final_warp

    # ###### Normal Calculation ######
    #
    # # TODO: Convert to global function in HelperFunctions:
    # def convert_to_uint8(image):
    #     max_value = np.max(image)
    #     min_value = np.min(image)
    #     return np.uint8(((image - min_value) / (np.max((max_value - min_value, 1)))) * 255)
    #
    # def get_projection_point_cloud(self):
    #     depth_image = self.camera_manager.get_depth(res_xy=(GlobalParams.DEPTH_IMAGE_SIZE.width, GlobalParams.DEPTH_IMAGE_SIZE.height))
    #     depth_map_adj = SingleImageCalibrator.convert_to_uint8(depth_image)
    #
    #     _, contours, hierarchy = cv2.findContours(depth_map_adj.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     # Create an output of all zeroes that has the same shape as the input
    #     # image
    #     out = np.zeros_like(depth_map_adj)
    #     # On this output, draw all of the contours that we have detected
    #     # in white, and set the thickness to be 3 pixels
    #     cv2.drawContours(out, contours, -1, 255, 5)
    #     # Spawn new windows that shows us the donut
    #     # (in grayscale) and the detected contour
    #
    #     out = np.array(out + 1, dtype = bool)
    #     depth_image = depth_image * out
    #
    #     print("max depth image: {}".format(depth_image.max()))
    #
    #     cropped_depth = cv2.bitwise_and(depth_image, depth_image, mask=self.mask)
    #
    #     if CALIBRATION_DEBUG:
    #         cv2.imwrite(CROPPED_DEPTH_IMAGE_PATH, cropped_depth)
    #
    #     pcl = PointCloud(cropped_depth, full_matrix=True)
    #     points = copy.deepcopy(pcl.point_cloud)
    #     points = points[points[:, 1].argsort(), :]
    #
    #     return points
    #
    # def find_plane_normal(self):
    #     bottom_left, bottom_right, top_left, top_right = self.get_display_corners_on_cam(
    #         needed_resolution=(GlobalParams.PROJECTOR_RESOLUTION.width, GlobalParams.PROJECTOR_RESOLUTION.height))
    #
    #     # TODO: Check the following to probably correct an error with the mask:
    #     # Maybe need to scale each param in tuple by GlobalParams.CALIBRATION_SCALE_FACTOR
    #
    #     bottom_left = (int(bottom_left[0][0][0]), int(bottom_left[0][0][1]))
    #     bottom_right = (int(bottom_right[0][0][0]), int(bottom_right[0][0][1]))
    #     top_left = (int(top_left[0][0][0]), int(top_left[0][0][1]))
    #     top_right = (int(top_right[0][0][0]), int(top_right[0][0][1]))
    #     contours = np.array([list(bottom_left), list(bottom_right), list(top_right), list(top_left)])
    #     img = np.zeros((GlobalParams.DEPTH_IMAGE_SIZE.height, GlobalParams.DEPTH_IMAGE_SIZE.width))
    #     cv2.fillPoly(img, pts=[contours], color=(255, 255, 255))
    #     self.mask = np.asanyarray(img, dtype='uint8')
    #
    #     self.find_normals_average(NORMAL_AVERAGING_ITERATIONS)
    #
    # def normalized_normal(self):
    #     normal, distance, _ = self.calculate_normal(self.get_projection_point_cloud())
    #     distance = distance / np.linalg.norm(normal)
    #     normal = normal / np.linalg.norm(normal)
    #     print("Normal: {}".format(normal))
    #     self.normals_array.append(np.array((normal[0], normal[1], normal[2], distance)))
    #
    # def calculate_normal(self, points):
    #     return Helper.get_best_plane_from_points(points)
    #
    # def find_normals_average(self, samples_count=100):
    #     self.normals_array = []
    #
    #     for i in range(samples_count):
    #         self.normalized_normal()
    #         stacked_normals = np.vstack(self.normals_array)
    #         mean_normal = np.mean(stacked_normals, axis=0)
    #         print("Sampling normal #{}".format(i))
    #
    #     stacked_normals = np.vstack(self.normals_array)
    #     mean_normal = np.mean(stacked_normals, axis=0)
    #     self.plane_normal = [mean_normal[0], mean_normal[1], mean_normal[2]]
    #     self.plane_distance = mean_normal[3]
    #
    #     # Enforce normal direction
    #     if self.plane_normal[2] < 0:
    #         self.plane_normal = np.multiply(self.plane_normal, -1)
    #         self.plane_distance = self.plane_distance * -1
    #
    #     # Normalize
    #     self.plane_normal = self.plane_normal / np.linalg.norm(self.plane_normal)
    #     self.plane_distance = self.plane_distance / np.linalg.norm(self.plane_normal)
    #
    #     print("Final Normal: {} Distance: {}".format(self.plane_normal, self.plane_distance))
    #
    # # Utility Functions
    #
    def transfrom_points_display_to_cam(self, pts):
        if len(pts) == 0:
            return np.array([])
        pt_transformed = np.squeeze(cv2.transform(np.float32(pts).reshape(-1, 1, 2),
                                                  np.linalg.inv(self.warp_mat_cam_2_displayed)), axis=1)
        return pt_transformed[:, :2].reshape((-1, 2)) / pt_transformed[:, 2].reshape((-1, 1))
    #
    # def get_display_corners_on_cam(self, needed_resolution=(1280, 800)):
    #     factor = self.screen_resolution.width / needed_resolution[0]
    #     top_right = np.expand_dims(self.transfrom_points_display_to_cam(np.array([[self.screen_resolution.width, 0]])), axis=1)
    #     top_left = np.expand_dims(self.transfrom_points_display_to_cam(np.array([[0, 0]])), axis=1)
    #     bottom_right = np.expand_dims(
    #         self.transfrom_points_display_to_cam(np.array([[self.screen_resolution.width, self.screen_resolution.height]])), axis=1)
    #     bottom_left = np.expand_dims(
    #         self.transfrom_points_display_to_cam(np.array([[0, self.screen_resolution.height]])), axis=1)
    #     return bottom_left / factor, bottom_right / factor, top_left / factor, top_right / factor
    #
    def transfrom_points_cam_to_display(self, pts):
        if len(pts) == 0 or pts[0] is None:
            return np.array([])
        pt_transformed = np.squeeze(cv2.transform(np.float32(pts).reshape(-1, 1, 2), self.warp_mat_cam_2_displayed),
                                    axis=1)
        return pt_transformed[:, :2].reshape((-1, 2)) / pt_transformed[:, 2].reshape((-1, 1))

class CalibrationFailed(Exception):
    def __init__(self, message=""):
        super().__init__(message)