import glob
import os
import shutil
import time
from sys import platform

import cv2
import numpy as np
from pyeyeengine.utilities.logging import Log
import pyeyeengine.utilities.global_params as Globals
from pyeyeengine.utilities.file_uploader import FileUploader
from pyeyeengine.camera_utils.frame_manager import FrameManager
from pyeyeengine.object_detection.shape_detector import ShapeDetector
from pyeyeengine.object_detection.table_detector import TableDetector

SAVE_MASK_DEBUG = True

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
TABLE_MASK_DEBUG_FOLDER = FILE_PATH + "/table_mask/"
GRID_IMAGES_SAVE_PATH = FILE_PATH + "/grid_images/"
TRANSFORMATION_FILE_NAME = "warp_mat_cam_2_displayed"
TRANSFORMATION_FILE_PATH_NO_EXT = os.path.join(FILE_PATH, TRANSFORMATION_FILE_NAME)
GRID_FORMAT = "chessboard"


# log = open("calibration.log", "a")
# sys.stdout = log


class AutoCalibrator:
    def __init__(self, screen_resolution=None, warp_mat_cam_2_displayed_path=TRANSFORMATION_FILE_PATH_NO_EXT + ".npy",
                 image_pairs_save_path=GRID_IMAGES_SAVE_PATH):
        self.manual_mask = False
        self.mask_edges = []
        self.calibrate_success = False
        self.screen_width, self.screen_height = screen_resolution if screen_resolution else self.get_screen_res()
        self.shapeDetector = ShapeDetector()
        self.tableDetector = TableDetector()
        self.image_pairs_save_path = image_pairs_save_path
        self.calibrator_pattern_collector = CalibrationPatternCollector((self.screen_width, self.screen_height),
                                                                        image_pairs_save_path=self.image_pairs_save_path)
        self.chessboard_detector = ChessboardCornerDetector(image_pairs_save_path=self.image_pairs_save_path)
        self.projection_flip_type = "none"
        self.table_mask = np.uint8(np.ones((960, 1280, 3)) * 255)  # entire field of view is default
        if os.path.isfile(warp_mat_cam_2_displayed_path):
            self.warp_mat_cam_2_displayed = np.load(warp_mat_cam_2_displayed_path)
        else:
            self.warp_mat_cam_2_displayed = np.eye(3)
        self.table_shape = None
        self.table_contour = None

    def calibrate(self, mode="table", recollect_imgs=True, screen_setter=None):
        Log.i("Starting Old Calibration", flow="calibration")
        if mode == "table" and not os.path.exists(FILE_PATH + "/table_data_manual.npz"):
            self.get_table_mask(screen_setter)
        else:
            Log.i("Manual mask found, will ignore auto table mask", flow="calibration")
        self.calibrator_pattern_collector.screen_setter = screen_setter
        self.get_matching_corners_and_find_homography(recollect_imgs)

    def get_table_mask(self, image_server=None):
        white_screen = self.generate_blurry_white_screen()
        if platform == "win32":
            cv2.namedWindow("white_background", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("white_background", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("white_background", white_screen)
            cv2.waitKey(1)
        elif image_server:
            image_server.set_image(white_screen)

        self.try_to_find_table()

        if platform == "win32":
            cv2.destroyWindow("white_background")

    def try_to_find_table(self):
        Log.i("Trying to detect table", flow="calibration")
        FileUploader.clean_local_folder()

        found = False

        for try_num in range(10):
            Log.d("Attempt #{} to find table".format(try_num), flow="calibration")
            FrameManager.getInstance().set_depth_resolution(Globals.DEPTH_HIGH_QUALITY)
            clean_depth = FrameManager.getInstance().get_depth_frame()
            depth_map = cv2.resize(clean_depth, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            FrameManager.getInstance().set_rgb_resolution(Globals.RGB_MEDIUM_QUALITY)
            rbg = FrameManager.getInstance().get_rgb_frame()
            FrameManager.getInstance().set_rgb_resolution(Globals.RGB_HIGH_QUALITY)
            rgb_high_res = FrameManager.getInstance().get_rgb_frame()

            tform_small_to_big = self.get_warp_ORB(rgb_high_res, rbg)

            # tform_small_to_big = self.get_warp_320x240_640x480(rgb_high_res, rbg)
            depth_map_tformed = cv2.warpPerspective(depth_map, np.linalg.inv(tform_small_to_big),
                                                    (depth_map.shape[1], depth_map.shape[0]), cv2.INTER_NEAREST)
            FileUploader.upload_image(depth_map_tformed, "depth_transformed.png")
            self.table_contour, self.table_shape = self.tableDetector.detect_table(depth_map_tformed, rgb_high_res,
                                                                                   save_path=FILE_PATH + "/")

            if self.table_contour is not None:
                self.table_contour /= 4
                self.table_mask = np.zeros_like(self.table_mask)
                self.tableDetector.draw_contours([np.int32(np.round(self.table_contour * 4))], rgb_high_res, "possible_mask.png")
                self.table_mask = cv2.drawContours(image=self.table_mask,
                                                   contours=[np.int32(np.round(self.table_contour * 4))],
                                                   contourIdx=0,
                                                   color=(255, 255, 255),
                                                   thickness=-1)
                np.savez(FILE_PATH + "/table_data", table_contour=self.table_contour, table_shape=self.table_shape,
                         table_mask=self.table_mask)
                Log.i("Auto Table Found", flow="calibration")
                found = True
                break  # if contour found use it

        cv2.imwrite(FILE_PATH + "/table_mask.png", self.table_mask)
        FileUploader.upload_image(self.table_mask, "table_mask.png")

        if not found:
            Log.i("Table Not Found", flow="calibration")

        if SAVE_MASK_DEBUG:
            FileUploader.upload_image(clean_depth, "cleanDepthMap.png")
            FileUploader.upload_image(depth_map, "depthMap.png")
            FileUploader.upload_image(rbg, "rgb.png")
            FileUploader.upload_image(rgb_high_res, "rgbHR.png")

    def load_table_data(self):

        if os.path.exists(FILE_PATH + "/table_data_manual.npz"):
            Log.i("Loading manual mask", flow="calibration")
            table_data = np.load(FILE_PATH + "/table_data_manual.npz", allow_pickle=True)
            self.manual_mask = True
            self.table_shape = str(table_data["table_shape"])
            self.table_mask = table_data["table_mask"]
            self.mask_edges = table_data["edge_params"]
            return "success"
        elif os.path.exists(FILE_PATH + "/table_data.npz"):
            Log.i("Loading auto mask", flow="calibration")
            table_data = np.load(FILE_PATH + "/table_data.npz", allow_pickle=True)
            self.table_contour = table_data["table_contour"]
            self.table_shape = str(table_data["table_shape"])
            self.table_mask = table_data["table_mask"]
            return "success"
        else:
            Log.e("Error loading table data, please rerun calibration!", flow="calibration", extra_details={"error":"file not found"})
            #raise CalibrationFailed("Can't load table data. No table data file. Please rerun calibration.")

    def generate_blurry_white_screen(self):
        white_screen = np.float32(np.zeros((self.screen_height, self.screen_width, 3)))
        white_screen[5:-5, 5:-5, :] = 1  # 5
        # filter = np.ones((51, 51))/ 51/ 51
        # dist_from_center_x = np.meshgrid(np.arange(-25,25))
        return np.uint8(cv2.filter2D(white_screen, -1, np.ones((21, 21)) / 21 / 21 * 255))  # 21

    def rgb_res_offset(self, rgb_large, rgb_small, table_contour, radius=4):
        rgb_small = cv2.resize(rgb_small, (rgb_large.shape[1], rgb_large.shape[0]))
        rgb_mean_reduced = (rgb_large.mean(axis=2)) - rgb_large.mean(axis=2).mean()
        # rgb_mean_reduced = cv2.resize(rgb_mean_reduced, (rgb_small.shape[1], rgb_small.shape[0]))
        rgb_small = (rgb_small.mean(axis=2)) - rgb_small.mean(axis=2).mean()
        rgb_small_cropped = rgb_small[radius:-radius, radius:-radius]

        corr = cv2.matchTemplate(np.float32(rgb_mean_reduced), np.float32(rgb_small_cropped), cv2.TM_CCORR_NORMED)
        offset = np.array(np.where(corr == corr.max())) - radius
        return np.fliplr(offset.reshape(1, -1))  # xy

    def get_warp_320x240_640x480(self, rgb_large, rgb_small, mask=None):
        rgb_small = cv2.resize(rgb_small, (rgb_large.shape[1], rgb_large.shape[0]))
        if mask is None:
            mask = np.ones((rgb_large.shape[0], rgb_large.shape[1]))
        rgb_mean_reduced = (rgb_large.mean(axis=2)) * np.uint8(mask > 0)  # - rgb_large.mean(axis=2).mean()
        rgb_small = (rgb_small.mean(axis=2)) * np.uint8(mask > 0)  # - rgb_small.mean(axis=2).mean()

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-12)
        (cc, tform_to_big_to_small) = cv2.findTransformECC(np.float32(rgb_mean_reduced), np.float32(rgb_small),
                                                           np.eye(3, 3, dtype=np.float32),
                                                           motionType=cv2.MOTION_HOMOGRAPHY, criteria=criteria)
        return tform_to_big_to_small

    def get_warp_ORB(self, rgb_large, rgb_small, mask=None):
        rgb_small = cv2.resize(rgb_small, (rgb_large.shape[1], rgb_large.shape[0]))
        if mask is None:
            mask = np.ones((rgb_large.shape[0], rgb_large.shape[1]))
        rgb_mean_reduced = (rgb_large.mean(axis=2)) * np.uint8(mask > 0)  # - rgb_large.mean(axis=2).mean()
        rgb_small = (rgb_small.mean(axis=2)) * np.uint8(mask > 0)  # - rgb_small.mean(axis=2).mean()

        MAX_FEATURES = 500
        GOOD_MATCH_PERCENT = 0.15
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(np.uint8(rgb_mean_reduced), None)
        keypoints2, descriptors2 = orb.detectAndCompute(np.uint8(rgb_small), None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        h_rigid = cv2.estimateRigidTransform(points1, points2, False)

        if not h_rigid is None:
            h = np.concatenate([h_rigid, np.array([0, 0, 1]).reshape((1, -1))], axis=0)
        else:
            h = np.eye(3)
        # print("h_rigid: ", file=open("./auto_calibrator_log.txt", "a"))
        # print(h, file=open("./auto_calibrator_log.txt", "a"))

        height, width, _ = rgb_large.shape
        warped_image = cv2.warpPerspective(rgb_large, h, (width, height))
        FileUploader.upload_image(warped_image, "warped_projection.png")

        return h

    def get_matching_corners_and_find_homography(self, recollect_imgs=True):
        if recollect_imgs:
            self.remove_all_files_in_path(GRID_IMAGES_SAVE_PATH)
            self.calibrator_pattern_collector.table_mask = self.table_mask
            self.calibrator_pattern_collector.collect_chessboard_pattern()

        if len(glob.glob(GRID_IMAGES_SAVE_PATH + "*.npz")) > 0:
            corners_cam_valid, corners_displayed_valid = \
                self.chessboard_detector.get_predetected_chessboard_from_npz()
        else:
            corners_cam_valid, corners_displayed_valid = \
                self.chessboard_detector.detect_chessboard_from_saved_images(self.table_mask)
        if corners_cam_valid is not None and corners_displayed_valid is not None:
            self.set_homography_from_matching_corners(corners_cam_valid, corners_displayed_valid)
        else:
            Log.e("Old Calibration Failed", flow="calibration", extra_details={"error": "did not detect any patterns"})
            # raise CalibrationFailed("Did not detect any chessboard patterns")

    def remove_all_files_in_path(self, path):
        files = glob.glob(path + '/*')

        try:
            for f in files:
                os.remove(f)
        except Exception as e:
            Log.w("File not found!")

    def set_homography_from_matching_corners(self, corners_cam, corners_displayed):
        if corners_cam.shape[0] >= 8 * 4:  # at least four detections of chessboard pattern
            flip_types = ['none', 'ud', 'lr', 'lr_ud']
            rand_inds = np.random.random_integers(0, corners_cam.shape[0] - 1, size=(32))
            trans_inliers = [cv2.findHomography(corners_cam[rand_inds, :], self.correct_corners_for_potential_flip(
                corners_displayed, flip_type=flip_type)[rand_inds, :], method=cv2.RANSAC, ransacReprojThreshold=7,
                                                maxIters=10000)
                             for flip_type in flip_types]  # method=cv2.LMEDS
            inliers_perc = np.array([set[1].mean() for set in trans_inliers])
            # warp_mat = trans_inliers[inliers_perc.argmax()][0]
            # print("inliers: ".format(inliers_perc))
            warp_mat, inliers_perc_final = cv2.findHomography(corners_cam, self.correct_corners_for_potential_flip(
                corners_displayed, flip_type=flip_types[inliers_perc.argmax()]), method=cv2.RANSAC,
                                                              ransacReprojThreshold=7,
                                                              maxIters=10000)
            if inliers_perc_final.mean() <= .5:
                Log.e("Old Calibration Failed", flow="calibration", extra_details={"error" : "did not detect patterns"})
                # raise CalibrationFailed('failed to calibrate - did not detect chessboard patterns properly')
            else:
                self.calibrate_success = True
                self.projection_flip_type = flip_types[inliers_perc.argmax()]
                Log.i('Old Calibration Successful', flow="calibration")
                self.warp_mat_cam_2_displayed = warp_mat
                np.save(TRANSFORMATION_FILE_PATH_NO_EXT, self.warp_mat_cam_2_displayed)
        else:
            Log.e("Old Calibration Failed", flow="calibration", extra_details={"error": "not enough patterns detected"})
            # raise CalibrationFailed(
            #     'failed to calibrate - unable to detect enough chessboard patterns, found only %d corners' %
            #     corners_cam.shape[0])

    def correct_corners_for_potential_flip(self, corners_displayed, flip_type='ud'):
        corners_displayed_corrected = corners_displayed.copy()
        if flip_type == 'ud' or flip_type == 'lr_ud':
            corners_displayed_corrected[0::2, :, 1] = corners_displayed[1::2, :, 1]
            corners_displayed_corrected[1::2, :, 1] = corners_displayed[0::2, :, 1]
        if flip_type == 'lr' or flip_type == 'lr_ud':
            corners_displayed_corrected[0::2, :, 0] = corners_displayed[1::2, :, 0]
            corners_displayed_corrected[1::2, :, 0] = corners_displayed[0::2, :, 0]
        return corners_displayed_corrected

    def get_screen_res(self):
        if platform == "darwin":
            print("Running on macOS")
            screen_width = 1280
            screen_height = 800
        elif platform == "linux" or platform == "linux2":
            print("Running on Linux")
            screen = os.popen("xrandr -q -d :0").readlines()[0]
            screen_width = int(screen.split()[7])
            screen_height = int(screen.split()[9][:-1])
        elif platform == "win32":
            print("Running on Windows")
            import tkinter
            root = tkinter.Tk()
            root.withdraw()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
        return screen_width, screen_height

    def set_warp_cam_to_display(self, warp):
        self.warp_mat_cam_2_displayed = warp

    def transfrom_points_cam_to_display(self, pts):
        if len(pts) == 0 or pts[0] is None:
            return np.array([])
        pt_transformed = np.squeeze(cv2.transform(np.float32(pts).reshape(-1, 1, 2), self.warp_mat_cam_2_displayed),
                                    axis=1)
        return pt_transformed[:, :2].reshape((-1, 2)) / pt_transformed[:, 2].reshape((-1, 1))

    def transfrom_points_display_to_cam(self, pts):
        if len(pts) == 0:
            return np.array([])
        pt_transformed = np.squeeze(cv2.transform(np.float32(pts).reshape(-1, 1, 2),
                                                  np.linalg.inv(self.warp_mat_cam_2_displayed)), axis=1)
        return pt_transformed[:, :2].reshape((-1, 2)) / pt_transformed[:, 2].reshape((-1, 1))

    def display_hands_with_homography(self, hands, mask):
        rbg = np.ones((self.screen_height, self.screen_width, 3)) * 255 * np.expand_dims(np.uint8(mask > 0), axis=2)
        for hand in hands:
            hand_transformed_to_display = cv2.transform(hand.pointing_finger_pt, self.warp_mat_cam_2_displayed)
            try:
                rbg = cv2.circle(rbg, (int(hand_transformed_to_display[0, 0, 0] / hand_transformed_to_display[0, 0, 2]),
                                       int(hand_transformed_to_display[0, 0, 1] / hand_transformed_to_display[
                                           0, 0, 2])),
                                 10, (0, 255, 0), -1)
            except:
                pass

        cv2.namedWindow("hands on screen", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("hands on screen", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)  # , cv2.WINDOW_FULLSCREEN)
        cv2.imshow('hands on screen', rbg)
        cv2.waitKey(1)

    def display_display_corners_on_cam(self, img):
        bottom_left, bottom_right, top_left, top_right = self.get_display_corners_on_cam()
        cv2.line(img, (top_right[0, 0, 0], top_right[0, 0, 1]), (top_left[0, 0, 0], top_left[0, 0, 1]), (255, 0, 0),
                 5)
        cv2.line(img, (top_right[0, 0, 0], top_right[0, 0, 1]), (bottom_right[0, 0, 0], bottom_right[0, 0, 1]),
                 (255, 0, 0), 5)
        cv2.line(img, (bottom_right[0, 0, 0], bottom_right[0, 0, 1]), (bottom_left[0, 0, 0], bottom_left[0, 0, 1]),
                 (255, 0, 0), 5)
        cv2.line(img, (bottom_left[0, 0, 0], bottom_left[0, 0, 1]), (top_left[0, 0, 0], top_left[0, 0, 1]),
                 (255, 0, 0), 5)
        cv2.imshow('rbg with contours', img)
        cv2.waitKey(1)

    def get_display_corners_on_cam(self):
        top_right = np.expand_dims(self.transfrom_points_display_to_cam(np.array([[self.screen_width, 0]])), axis=1)
        top_left = np.expand_dims(self.transfrom_points_display_to_cam(np.array([[0, 0]])), axis=1)
        bottom_right = np.expand_dims(
            self.transfrom_points_display_to_cam(np.array([[self.screen_width, self.screen_height]])), axis=1)
        bottom_left = np.expand_dims(
            self.transfrom_points_display_to_cam(np.array([[0, self.screen_height]])), axis=1)
        return bottom_left, bottom_right, top_left, top_right

    def can_camera_see_whole_screen(self):
        bottom_left, bottom_right, top_left, top_right = self.get_display_corners_on_cam()
        return (bottom_left > 0).all() and (bottom_right > 0).all() and (top_left > 0).all() and (top_right > 0).all() \
               and bottom_left[0, 0, 1] < 320 and \
               top_left[0, 0, 1] < 320 and bottom_right[0, 0, 0] < 240 and top_right[0, 0, 0] < 240


class CalibrationFailed(Exception):
    def __init__(self, message=""):
        super().__init__(message)


class CalibrationPatternCollector:
    def __init__(self, screen_resolution, image_pairs_save_path=GRID_IMAGES_SAVE_PATH,
                 chessboard_grid_gaps=0.15, screen_setter=None, save_images=False):
        self.CHESSBOARD_GRID = np.arange(.05, .95, chessboard_grid_gaps)
        self.screen_width, self.screen_height = screen_resolution
        self.image_pairs_save_path = image_pairs_save_path
        self.make_sure_save_folder_exists()
        self.screen_setter = screen_setter
        self._save_images = save_images
        self.num_blocks_per_chessboard = 5
        self.table_mask = 1
        self.next_grid_image = None

    def make_sure_save_folder_exists(self):
        if not os.path.isdir(self.image_pairs_save_path):
            os.mkdir(self.image_pairs_save_path)

    def save_chessboard_pattern(self):
        shutil.rmtree(self.image_pairs_save_path)
        self.make_sure_save_folder_exists()
        [self._save_grid_images(self._create_local_chessboard_img(np.array([i, j])),
                                i, j, type="displayed")
         for i in np.nditer(self.CHESSBOARD_GRID)
         for j in np.nditer(self.CHESSBOARD_GRID)]
        return self.image_pairs_save_path

    def save_chessboard_patters_pngs(self):
        return [self._save_grid_images(
            self._create_local_chessboard_img(np.array([i, j])), i, j, type="displayed")
            for i in np.nditer(self.CHESSBOARD_GRID)
            for j in np.nditer(self.CHESSBOARD_GRID)]

    def depricated_get_chessboard_patters_pngs(self):
        return [cv2.imencode('.png', self._create_local_chessboard_img(np.array([i, j])))[1].tobytes()
                for i in np.nditer(self.CHESSBOARD_GRID)
                for j in np.nditer(self.CHESSBOARD_GRID)]

    def collect_chessboard_pattern(self):
        if self.screen_setter is None:
            cv2.namedWindow("chessboard_fullscreen", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("chessboard_fullscreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._collect_chessboard_with_running_window()
        if self.screen_setter is None:
            cv2.destroyWindow('chessboard_fullscreen')

    def _collect_chessboard_with_running_window(self):
        start_time = time.clock()
        ij_s = [[i, j]
                for i in np.nditer(self.CHESSBOARD_GRID)
                for j in np.nditer(self.CHESSBOARD_GRID)]
        next_ij_s = ij_s[:-1] + [[None, None]]
        self.screen_setter.set_image(self._create_chessboard_grid(3, 3, block_size=31) * 255)
        [self._display_grid_and_save(ij[0], ij[1], next_ij[0], next_ij[1]) for ij, next_ij in zip(ij_s, next_ij_s)]
        # print("[CHESSBOARD] show + capture all images: {}".format((time.clock() - start_time)))

    def _display_grid_and_save(self, percent_i, percent_j, percent_i_next=None, percent_j_next=None):
        grid_img_displayed = self._display_gird(percent_i, percent_j, percent_i_next, percent_j_next)
        # self._save_grid_images(grid_img_displayed, percent_i, percent_j, type="displayed")
        FrameManager.getInstance().set_rgb_resolution(Globals.SCREEN_RESOLUTION)
        rgb_with_table_mask = FrameManager.getInstance().get_rgb_frame(mask=self.table_mask > 0)
        if self._save_images:
            self._save_grid_images(grid_img_displayed, percent_i, percent_j, type="displayed")
            self._save_grid_images(rgb_with_table_mask, percent_i, percent_j, type="viewed")
        else:
            self._find_corners_and_save(grid_img_displayed.copy(), rgb_with_table_mask.copy(),
                                        percent_i.copy(), percent_j.copy())
            # threading.Thread(target=self._find_corners_and_save,
            #                  args=(grid_img_displayed.copy(), rgb_with_table_mask.copy(),
            #                        percent_i.copy(), percent_j.copy()), daemon=True).start()
            # self._find_corners_and_save(grid_img_displayed, rgb_with_table_mask, percent_i, percent_j)

    def _save_grid_images(self, img, percent_i, percent_j, add_label=False, type=""):
        if add_label:
            img = cv2.putText(img, "%d, %d " % (percent_i * 100, percent_j * 100),
                              (50, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (0, 0, 0), thickness=3)
        start_time = time.clock()

        cv2.imwrite(GRID_IMAGES_SAVE_PATH + "/" + type + "%d_%d.png" % (percent_i * 100, percent_j * 100), img)
        # print("[CHESSBOARD] _save_grid_images: {}".format((time.clock() - start_time)))

    def _find_corners_and_save(self, grid_img_displayed, rgb_with_table_mask, percent_i, percent_j):
        ret, corners_cam_curr, corners_displayed_curr = \
            ChessboardCornerDetector()._detect_chessboard(rgb_with_table_mask, grid_img_displayed)
        np.savez(GRID_IMAGES_SAVE_PATH + "/%d_%d" % (percent_i * 100, percent_j * 100),
                 ret=ret, corners_cam_curr=corners_cam_curr, corners_displayed_curr=corners_displayed_curr)
        self._save_grid_images(rgb_with_table_mask, percent_i, percent_j, type="viewed")
        self._save_grid_images(grid_img_displayed, percent_i, percent_j, type="displayed")

    def _display_gird(self, percent_i, percent_j, percent_i_next=None, percent_j_next=None):
        # start_time_total = time.clock()
        if self.screen_setter is None:
            grid_img_displayed = self._create_local_chessboard_img(np.array([percent_i, percent_j]))
            cv2.imshow('chessboard_fullscreen', grid_img_displayed)
            cv2.waitKey(1)
            time.sleep(.4)
        else:
            grid, pad_x_after, pad_x_before, pad_y_after, \
            pad_y_before, block_size, num_blocks = self.get_chessboard_params(np.array([percent_i, percent_j]))
            # self.screen_setter.set_chessboard_image(num_blocks, block_size, pad_y_before, pad_x_before)
            self.screen_setter.set_image_top_left(top=pad_y_before, left=pad_x_before)
            grid_img_displayed = self.chessboard_params_to_image(grid, pad_x_after, pad_x_before, pad_y_after,
                                                                 pad_y_before)
            # print("[CHESSBOARD] X:{} Y:{}".format(pad_y_before, pad_x_before))
            time.sleep(.8)
        # print("time_to_gen_and_show_grid_image: %f" % (time.clock() - start_time_total))
        return grid_img_displayed

    def _create_local_chessboard_img(self, location_perc_screen_xy=np.array([.5, .5])):
        grid, pad_x_after, pad_x_before, pad_y_after, \
        pad_y_before, block_size, num_blocks = self.get_chessboard_params(location_perc_screen_xy,
                                                                          num_blocks=self.num_blocks_per_chessboard)
        return self.chessboard_params_to_image(grid, pad_x_after, pad_x_before, pad_y_after, pad_y_before)

    def chessboard_params_to_image(self, grid, pad_x_after, pad_x_before, pad_y_after, pad_y_before):
        # print("chessboard_params_to_image %f", time.clock(), file=open("./auto_calibration_log.txt", "a"))
        return np.uint8(np.lib.pad(grid, ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
                                   'maximum'))[:self.screen_height, :self.screen_width]

    def get_chessboard_params(self, location_perc_screen_xy, block_size=31, num_blocks=5):
        width = self.screen_width
        height = self.screen_height
        num_black_blocks = int(num_blocks / 2 + 1)
        grid = self._create_chessboard_grid(num_black_blocks, num_black_blocks, block_size=block_size) \
            if GRID_FORMAT == "chessboard" \
            else self._create_circle_grid(num_black_blocks, num_black_blocks, block_size=block_size)
        pad_x_before = int(np.floor((width - grid.shape[1]) * location_perc_screen_xy[0]) + 1)
        pad_x_after = int(np.floor((width - grid.shape[1]) * (1 - location_perc_screen_xy[0])))
        pad_y_before = int(np.floor((height - grid.shape[0]) * location_perc_screen_xy[1]) + 1)
        pad_y_after = int(np.floor((height - grid.shape[0]) * (1 - location_perc_screen_xy[1])))
        return grid, pad_x_after, pad_x_before, pad_y_after, pad_y_before, block_size, num_blocks

    def _create_chessboard_grid(self, num_blocks_height, num_blocks_width, block_size=75):
        return np.tile(np.array([[0, 255], [255, 0]]).repeat(block_size, axis=0).repeat(block_size, axis=1),
                       (num_blocks_height, num_blocks_width))[:-block_size, :-block_size]

    def _create_circle_grid(self, num_blocks_height, num_blocks_width, block_size=75):
        white_square = np.ones((block_size, block_size), dtype=np.uint8) * 255
        one_circle = cv2.circle(white_square.copy(), (int(block_size / 2), int(block_size / 2)), int(block_size / 3),
                                color=(0, 0, 0), thickness=-1)
        grid = np.concatenate([np.concatenate([one_circle, white_square], axis=1),
                               np.concatenate([white_square, one_circle, ], axis=1)], axis=0)
        return np.tile(grid, (num_blocks_height, num_blocks_width))  # [:-block_size, :-block_size]


class ChessboardCornerDetector:
    def __init__(self, image_pairs_save_path=GRID_IMAGES_SAVE_PATH):
        self.image_pairs_save_path = image_pairs_save_path

    def get_predetected_chessboard_from_npz(self):
        corners_cam_valid, corners_displayed_valid = [], []
        chessboard_corners_files = glob.glob(GRID_IMAGES_SAVE_PATH + "*.npz")
        for file in chessboard_corners_files:
            corners_curr = np.load(file)
            if corners_curr['ret']:
                corners_cam_valid.append(corners_curr['corners_cam_curr'])
                corners_displayed_valid.append(corners_curr['corners_displayed_curr'])
        if len(corners_cam_valid) == 0:
            corners_cam_valid, corners_displayed_valid = None, None
        else:
            corners_cam_valid = np.concatenate(corners_cam_valid, axis=0)
            corners_displayed_valid = np.concatenate(corners_displayed_valid, axis=0)
        return corners_cam_valid, corners_displayed_valid

    def detect_chessboard_from_saved_images(self, cam_mask):
        start_time = time.clock()
        windows = [self._detect_chessboard_in_image_pairs(image_name, cam_mask)
                   for image_name in
                   self._get_unique_image_names_from_paths(glob.glob(self.image_pairs_save_path + "*.png"))]
        valid_windows = [window[1] for window in windows if window[0]]
        if len(valid_windows) > 0:
            corners_cam_valid = np.concatenate(valid_windows, axis=0)
            corners_displayed_valid = np.concatenate([window[2] for window in windows if window[0]], axis=0)
        else:
            corners_cam_valid = None
            corners_displayed_valid = None
        elapsed_time = time.clock() - start_time
        # print("finding corners in all images took : ", elapsed_time, " (s)")
        return corners_cam_valid, corners_displayed_valid

    def _get_unique_image_names_from_paths(self, image_paths):
        return list(set([os.path.splitext(path)[0].split("displayed")[-1].split("viewed")[-1] for path in image_paths]))

    def _detect_chessboard_in_image_pairs(self, image_name, cam_mask):
        grid_img_displayed = self._read_grid_images(image_name, type="displayed")
        rgb_with_table_mask = self._read_grid_images(image_name, type="viewed") * np.uint8(cam_mask > 0)
        return self._detect_chessboard(rgb_with_table_mask, grid_img_displayed)

    def _read_grid_images(self, image_name, type=""):
        return cv2.imread(self.image_pairs_save_path + type + "%s.png" % (image_name))

    def _detect_chessboard(self, rbg_with_table_mask, grid_img_displayed):
        ret_cam, corners_cam_curr = self._detect_in_single_img(rbg_with_table_mask, 4, 4)
        if ret_cam:
            corners_cam_curr /= 4
        ret_displayed, corners_displayed_curr = self._detect_in_single_img(grid_img_displayed, 4, 4)
        return ret_cam and ret_displayed, corners_cam_curr, corners_displayed_curr

    def _detect_in_single_img(self, img, num_corners_x, num_corners_y):
        if len(img.shape) > 2:
            img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
        # cv2.imshow("cam", img)
        # cv2.waitKey(0)
        if GRID_FORMAT == "chessboard":
            # ret, corners = cv2.findChessboardCorners(img, (num_corners_x, num_corners_y), None)
            ret, corners = self.find_chessboard_corners_fast(img, (num_corners_x, num_corners_y))
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), criteria)
                # print("corners: ")
                # print(np.sort(corners,axis=0))
                # print("corners subpixel: ")
                # print(np.sort(corners2, axis=0))
        else:
            # ret, corners = cv2.findCirclesGrid(img, (num_corners_x, num_corners_y), None)
            ret, corners = CircleGridDetector().detect_circle_grid(img, (num_corners_x, num_corners_y))
        if ret:
            corners = self._sort_4_corners(corners)
            # corners = self._sort_by_closest_to_top_left(corners)
        return ret, corners

    def _sort_4_corners(self, corners):
        dist_to_center = np.power(corners - corners.mean(axis=0).reshape(1, 1, 2), 2).sum(axis=2)
        sorted_inds = np.argsort(dist_to_center, axis=0).reshape(-1)
        return np.concatenate([self._sort_by_closest_to_top_left(corners[sorted_inds[:4], :, :]),
                               self._sort_by_closest_to_top_left(corners)], axis=0)
        # return self.sort_by_closest_to_top_left(corners[sorted_inds[:4], :, :])
        # return np.concatenate([self.sort_by_closest_to_top_left(corners[sorted_inds[:4], :, :]),
        #                       self.sort_by_closest_to_top_right(corners[sorted_inds[:4], :, :])], axis=0)

    def _sort_by_closest_to_top_left(self, corners):
        top_left = corners.min(axis=0)
        dist_to_top_left = np.power(corners - top_left.reshape(1, 1, 2), 2).sum(axis=2)
        sorted_inds = np.argsort(dist_to_top_left, axis=0).reshape(-1)
        return corners[sorted_inds[[0, -1]], :, :]

    def _sort_by_closest_to_top_right(self, corners):
        top_right = np.array([corners.min(axis=0)[0, 1], corners.max(axis=0)[0, 0]])
        dist_to_top_right = np.power(corners - top_right.reshape(1, 1, 2), 2).sum(axis=2)
        sorted_inds = np.argsort(dist_to_top_right, axis=0).reshape(-1)
        return corners[sorted_inds[[0, -1]], :, :]

    def find_chessboard_corners_fast(self, img, num_corners_xy, reduction_factor=3):
        ret, corners = cv2.findChessboardCorners(img[::reduction_factor, ::reduction_factor], num_corners_xy, None)
        if ret:
            corners = corners * reduction_factor
            min_xy = np.int32(np.maximum(np.squeeze(corners, axis=1).min(axis=0) - 100, 0))
            max_xy = np.int32(np.squeeze(corners, axis=1).max(axis=0) + 100)
            max_xy[0] = np.minimum(max_xy[0], img.shape[1])
            max_xy[1] = np.minimum(max_xy[1], img.shape[0])
            cropped_image = img[min_xy[1]: max_xy[1], min_xy[0]: max_xy[0]]
            ret, corners = cv2.findChessboardCorners(cropped_image, num_corners_xy, None)
            if ret:
                corners[:, 0, 0] += min_xy[0] + 1
                corners[:, 0, 1] += min_xy[1] + 1
        return ret, corners


class CircleGridDetector():
    def __init__(self):
        blobParams = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        blobParams.minThreshold = 8
        blobParams.maxThreshold = 255

        # Filter by Area.
        blobParams.filterByArea = True
        blobParams.minArea = 10  # minArea may be adjusted to suit for your experiment
        blobParams.maxArea = 2500  # maxArea may be adjusted to suit for your experiment

        # Filter by Circularity
        blobParams.filterByCircularity = True
        blobParams.minCircularity = 0.1

        # Filter by Convexity
        blobParams.filterByConvexity = True
        blobParams.minConvexity = 0.87

        # Filter by Inertia
        blobParams.filterByInertia = True
        blobParams.minInertiaRatio = 0.01

        # Create a detector with the parameters
        self.blobDetector = cv2.SimpleBlobDetector_create(blobParams)

    def detect_circle_grid(self, img, grid_shape=(5, 5)):
        # key_points = self.blobDetector.detect(img)
        # im_with_keypoints = cv2.drawKeypoints(np.tile(np.expand_dims(np.ones_like(img)*255,axis=2),(1,1,3)),
        #                                       key_points, np.array([]), (0, 255, 0),
        #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # # im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
        _, centroids = cv2.findCirclesGrid(img, grid_shape, None,
                                           flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=self.blobDetector)
        ret = centroids is not None and (centroids.shape[0] == ((grid_shape[0] - 1) * (grid_shape[1] - 1)))
        return ret, centroids
