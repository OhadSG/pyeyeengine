import inspect
import os
import time
import random

import numpy as np
import cv2
from pyeyeengine.eye_engine.ransac_utils import fit_line, fit_line_weighted
from pyeyeengine.utilities.file_uploader import FileUploader
import pyeyeengine.utilities.global_params as Globals
from pyeyeengine.camera_utils.frame_manager import FrameManager
from pyeyeengine.utilities.logging import Log

import json

path_to_folder, file_name = os.path.split(os.path.realpath(__file__))
with open(path_to_folder + '/table_detector_config.json') as f:
    config = json.load(f)

MARGIN_OF_ERROR = config.get('ROLL_EDGES_IN', .5)
EDGES_DEPTH_BLUR_KERNEL = (11, 11)
DEPTH_CONTOUR_DILATION = EDGES_DEPTH_BLUR_KERNEL[0]
RGB_EDGES_THRESHOLD = 50
MIN_TABLE_CONTOUR_AREA = config.get('MIN_TABLE_CONTOUR_AREA', 32000)
MIN_TABLE_CONTOUR_SIDE_LENGTH = config.get('MIN_TABLE_CONTOUR_SIDE_LENGTH', 400)
BASE_HEIGHT = 2000

# notes:
# table expected to :
# 1) be a large flat surface
# 2) be near the center of the cameras view
# 3) cover no more than 75% of the field of view of the camera
# 4) be circular, rectangular, or square
from pyeyeengine.object_detection.shape_detector import ShapeDetector

class TableDetectionFailed(Exception):
    def __init__(self, message=""):
        super().__init__(message)

class TableDetector:
    def __init__(self):
        self.shapeDetector = ShapeDetector()
        self.edges = []
        self.image_index = 0
        pass

    def random_color():
        return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    def draw_contours(self, contours, image, file_name):
        for contour in contours:
            color = TableDetector.random_color()
            rounded_contour = self.round_int32(contour)
            # Log.d("Passed to cv2.drawContours: {} {} {}".format(image, rounded_contour, color))
            try:
                if rounded_contour is not None:
                    image = cv2.drawContours(image, [rounded_contour], 0, color, 1)
            except Exception as e:
                Log.e("Error drawing contours", extra_details={"error": "{}".format(e)})
        FileUploader.save_image(image, file_name, folder="table_detection/")

    def detect_table(self, depth_map, rgb, display=True, save_path=None):
        clean_rgb = rgb.copy()
        contours = self.find_potential_table_contours(depth_map)
        #Log.d("FOUND CONTOURS: {}".format(len(contours)))
        self.draw_contours(contours, rgb.copy(), "found_contours.png")
        table_contour = self.decide_which_contour_is_the_table(contours, depth_map)
        self.draw_contours([self.round_int32(table_contour)], rgb.copy(), "possible_table_contour.png")
        # Log.d("Table contour: {}".format(table_contour))
        if save_path is not None and table_contour is not None:
            rgb_rough_contour = cv2.drawContours(rgb.copy(), [self.round_int32(table_contour)], 0, (255, 0, 255), 1)
            cv2.imwrite(save_path + "table_detected_rough.png", rgb_rough_contour)
            FileUploader.save_image(rgb_rough_contour, "rough_table_contour.png", folder="table_detection/")

        table_shape = 'None'
        if table_contour is not None:
            # undo edges dilation used to clearly seperate objects
            if self.shapeDetector.detect(table_contour) == 'circle':
                # table_contour = self.dilate_contour(table_contour, DEPTH_CONTOUR_DILATION*2, shape='circle')
                table_contour = self.dilate_ellipse_contour(table_contour, depth_map=depth_map,
                                                            dilate_amount=DEPTH_CONTOUR_DILATION*2)
                # circles for some reason require more dilation
                table_contour = self.optimize_cricle_contour_RANSAC(self.round_int32(table_contour), rgb)
                table_contour = self.dilate_contour(table_contour, -MARGIN_OF_ERROR, shape='circle')
                table_shape = 'circle'
                Log.d("Found circle table")
            elif self.shapeDetector.detect(table_contour) == 'rectangle':

                table_contour = self.dilate_rect_contour(table_contour, DEPTH_CONTOUR_DILATION*2)
                FileUploader.save_image(self.draw_table_on_rbg(clean_rgb.copy(), [self.round_int32(table_contour)]),
                                        "contour_1.png", folder="table_detection/")
                table_contour = self.optimize_rectangle_contour_RANSAC(self.round_int32(table_contour), rgb, depth_map)
                FileUploader.save_image(self.draw_table_on_rbg(clean_rgb.copy(), [self.round_int32(table_contour)]),
                                        "contour_2.png", folder="table_detection/")
                table_contour = self.dilate_rect_contour(table_contour, -MARGIN_OF_ERROR)
                FileUploader.save_image(self.draw_table_on_rbg(clean_rgb.copy(), [self.round_int32(table_contour)]),
                                        "contour_3.png", folder="table_detection/")
                table_shape = 'rectangle'
                Log.d("Found rectangle table")
        # else:
        #     raise TableDetectionFailed('No table of round or rectangular shape found.')

        if display:
            self.display(rgb.mean(axis=2), [self.round_int32(table_contour)])
            # self.display(self.edges * 255, [self.round_int32(table_contour)], self.edges * 255)

        if save_path is not None:
            cv2.imwrite(save_path + "table_detected.png",
                        self.draw_table_on_rbg(rgb, [self.round_int32(table_contour)]))
            FileUploader.save_image(self.draw_table_on_rbg(rgb, [self.round_int32(table_contour)]), "table_detected.png", folder="table_detection/")

        return table_contour, table_shape

    def find_potential_table_contours(self, depth_map, display=False, rgb=None):
        self.edges = (self.canny_override(depth_map, 150, 2500, kernel_size=5, factor=25) +
                      cv2.resize(np.float32(self.canny_override(depth_map[2::4, 2::4],
                                                                75, 2500, kernel_size=5, factor=25)),
                                 (depth_map.shape[1], depth_map.shape[0]))) > 0

        self.edges = np.uint8(
            cv2.filter2D(np.uint8(self.edges), -1, np.ones(EDGES_DEPTH_BLUR_KERNEL)) * self.edges) * 255
        FileUploader.save_image(self.edges, "found_edges.png", folder="table_detection/")
        edges_dilated = cv2.dilate(self.edges, None, iterations=DEPTH_CONTOUR_DILATION)
        FileUploader.save_image(edges_dilated, "dilated_edges.png", folder="table_detection/")
        # cv2.imshow('edges_dilated', cv2.resize(edges_dilated, (0, 0), fx=.5, fy=.5))
        # cv2.imshow('edges_dilated', cv2.resize(edges_dilated, (0, 0), fx=.5, fy=.5))
        # cv2.waitKey(0)

        # vel = self.get_laplacian(2, depth_map, 5)
        # accel = self.get_laplacian(2, vel, 5)
        # cv2.imshow("accel", ntp.uint8(accel / 15))
        # cv2.waitKey(0)

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import axes3d
        #
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # x, y = np.meshgrid(np.linspace(0, depth_map.shape[1] - 1, depth_map.shape[1]),
        #                    np.linspace(0, depth_map.shape[0] - 1, depth_map.shape[0]))
        # z = depth_map
        # ax.scatter(np.int32(x[::20, ::20]), np.int32(y[::20, ::20]),
        #            z[np.int32(y[::20, ::20]), np.int32(x[::20, ::20])])
        # plt.show()

        n_componenets, cc_map = cv2.connectedComponents(np.uint8(edges_dilated == 0) * 255)
        # cv2.imshow("cc_map", np.uint8((cc_map/n_componenets)*255))
        # cv2.waitKey(0)
        FileUploader.save_image(cc_map, "connected_components.png", folder="table_detection/")
        contours = [cv2.findContours(np.uint8(cc_map == idx), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1][-1]
                    for idx in range(1, n_componenets)]

        # _, contours, hierarchy = cv2.findContours(np.uint8(edges_dilated == 0) * 255, cv2.RETR_CCOMP,
        #                                           cv2.CHAIN_APPROX_NONE)

        if display:
            self.display(depth_map, contours, edges_dilated)
        return contours

    def decide_which_contour_is_the_table(self, contours, depth_map):
        contours_big_enough = [contour for contour in contours
                               if cv2.contourArea(contour) >
                               MIN_TABLE_CONTOUR_AREA / np.maximum(self.median_contour_height(contour, depth_map),
                                                                   1) * BASE_HEIGHT]

        contours_big_enough = [contour for contour in contours_big_enough if
                               np.array(cv2.boundingRect(contour))[2:].min() >
                               MIN_TABLE_CONTOUR_SIDE_LENGTH /
                               np.maximum(self.median_contour_height(contour, depth_map), 1) * BASE_HEIGHT]
        table_shaped_contours = [contour for contour in contours_big_enough if
                                 (self.shapeDetector.detect(contour) == 'rectangle' or self.shapeDetector.detect(
                                     contour) == 'circle')]
        not_the_whole_screen_contours = [contour for contour in table_shaped_contours if
                                         cv2.contourArea(contour) < np.array(depth_map.shape[:2]).prod() * .5]
        screen_center = np.array(depth_map.shape[:2]) / 2
        dist_from_center_of_screen = [self.shapeDetector.find_dist_from_closest_pt_in_contour_to_a_pt(
            contour, int(screen_center[0]), int(screen_center[1])) *
                                      np.power(cv2.contourArea(contour) + 1, 1 / 4)
                                      for contour in not_the_whole_screen_contours]
        if len(dist_from_center_of_screen) > 0:
            table_contour = not_the_whole_screen_contours[np.array(dist_from_center_of_screen).argmin()]
        else:
            table_contour = None
        return table_contour

    def median_contour_height(self, contour, depth_map):
        x, y = contour[:, 0, 0], contour[:, 0, 1]
        return np.median(depth_map[y, x])

    def optimize_rectangle_contour_RANSAC(self, table_contour, rgb, depth):
        table_contour_original = table_contour.copy()
        rect = cv2.minAreaRect(table_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = self.rectangle_to_trapezoid(np.squeeze(table_contour, axis=1), np.ones((table_contour.shape[0])),
                                          box, line_dist_thresh=21)
        rect_only, contour_only = np.zeros_like(rgb), np.zeros_like(rgb)
        # rect_only = cv2.drawContours(rect_only, [box], 0, (255, 255, 255), 5)
        rect_only = cv2.drawContours(rect_only, [table_contour], 0, (255, 255, 255), 31)

        rect_only = rect_only[:, :, 0]

        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rgb_edges = np.uint8(self.canny_override(rgb_gray, RGB_EDGES_THRESHOLD)) * 255
        rgb_edges[[0, -1], :] = 255
        rgb_edges[:, [0, -1]] = 255
        #
        # rgb_to_show_r, rgb_to_show_g, rgb_to_show_b = rgb[:, :, 0].copy(), rgb[:, :, 1].copy(), rgb[:, :, 2].copy()
        # rgb_to_show_r[rgb_edges > 0] = 0
        # rgb_to_show_g[rgb_edges > 0] = 255
        # rgb_to_show_b[rgb_edges > 0] = 255
        # # rgb_to_show_g[rect_only > 0] = 0
        # # rgb_to_show_b[rect_only > 0] = 0
        # rgb_to_show = np.concatenate([rgb_to_show_r[:, :, np.newaxis], rgb_to_show_g[:, :, np.newaxis],
        #                               rgb_to_show_b[:, :, np.newaxis]], axis=2)
        # cv2.imshow("rgb_edges_rect_only", rgb_to_show)
        # cv2.waitKey(0)

        rgb_edges_toshow = rgb.copy()
        # rgb_edges_toshow[np.tile(np.expand_dims(rgb_edges, axis=2), (1, 1, 3)) > 0] = 0

        # cv2.imshow("local_max", cv2.resize(local_max, (320,240)))
        # cv2.waitKey(1)

        # relevant_edges = (cv2.dilate(rgb_edges * rect_only, None, None) * local_max) > 0
        relevant_edges = rgb_edges * np.uint8(rect_only > 0)  # (np.float32(local_max) * np.float32(rect_only)) > 0
        self.edges = np.uint8(relevant_edges)

        relevant_pts = np.fliplr(np.array(np.where(relevant_edges)).T)
        gradient_at_relevant_pts = self.get_laplacian(1, rgb_gray, 3)[relevant_edges > 0]
        corr_coeffs = []
        rects = []
        bboxes = []
        for _ in range(1):
            # random_pts = np.expand_dims(relevant_pts[np.random.choice(relevant_pts.shape[0],
            #                                                           int(relevant_pts.shape[0] / 1), replace=False),
            #                             :], axis=1)
            # rect = cv2.minAreaRect(random_pts)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # box = self.rectangle_to_trapezoid(relevant_pts, gradient_at_relevant_pts, box, line_dist_thresh=21)
            new_box = self.rectangle_to_trapezoid(relevant_pts, gradient_at_relevant_pts, box, line_dist_thresh=21)

            if new_box is not None:
                box = new_box

            bboxes.append(box)
            rect_only = np.zeros_like(rgb)
            rect_only = cv2.drawContours(rect_only, [self.round_int32(box)], 0, (255, 255, 255), 1)[:, :, 0]
            rects.append(rect)
            # corr_coeffs.append((rect_only * rgb_edges).sum())
            corr_coeffs.append((rect_only * self.edges).sum())

        box = bboxes[np.array(corr_coeffs).argmax()]
        table_contour = np.expand_dims(self.sort_pts_clockwise(box), axis=1)
        rgb_edges_toshow = cv2.drawContours(rgb_edges_toshow, [np.int32(np.round(table_contour))], 0, (255, 0, 0), 1)
        rgb_edges_toshow = cv2.drawContours(rgb_edges_toshow, [np.int32(np.round(table_contour_original))], 0,
                                            (0, 255, 0), 1)
        cv2.imwrite("./edges.png", rgb_edges_toshow)
        FileUploader.save_image(rgb_edges_toshow, "rgb_edges.png", folder="table_detection/")
        table_contour = self.corners_to_dense_contour(table_contour)
        return np.float32(table_contour)  # -.5# edges locations are rounded up...

    def corners_to_dense_contour(self, corners):
        final_contour = []
        for corner_idx in range(4):
            pt1 = corners[corner_idx, 0, :]
            pt2 = corners[(corner_idx + 1) % 4, 0, :]
            x = np.linspace(pt1[0], pt2[0], 10)
            y = np.linspace(pt1[1], pt2[1], 10)
            final_contour.append(np.stack([x, y], axis=1))
        return np.expand_dims(self.sort_pts_clockwise(np.concatenate(final_contour, axis=0)), axis=1)

    def rectangle_to_trapezoid(self, edge_pts, weights, rect_corner_pts, line_dist_thresh=11):
        for idx in range(4):
            weights = weights[np.linalg.norm(edge_pts - rect_corner_pts[idx, :].reshape(1, 2), axis=1) > 22]
            edge_pts = edge_pts[np.linalg.norm(edge_pts - rect_corner_pts[idx, :].reshape(1, 2), axis=1) > 22, :]

        rect_corner_pts = np.float32(self.sort_pts_clockwise(rect_corner_pts))
        lines = []
        for idx in range(4):
            pt1 = rect_corner_pts[idx, :]
            pt2 = rect_corner_pts[(idx + 1) % 4, :]
            dist_to_line = self.distance_two_pts_line_to_pts(pt1, pt2, edge_pts)
            relevant_pts = edge_pts[dist_to_line <= line_dist_thresh, :]
            relevant_weights = weights[dist_to_line <= line_dist_thresh]

            # import matplotlib as mpl
            # mpl.use('Agg')
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax1 = fig.add_subplot(111)
            # ax1.scatter(edge_pts[:, 0], edge_pts[:, 1], s=5, c='b', marker="s", label='first')
            # ax1.scatter(relevant_pts[:, 0], relevant_pts[:, 1], s=5, c='r', marker="o", label='second')
            # ax1.plot(np.array([pt1[0], pt2[0]]), np.array([pt1[1], pt2[1]]), "g-")
            # plt.legend(loc='upper left')
            # fig.savefig('./temp.png')

            if len(relevant_pts) >= 20:
                # line = np.poly1d(np.polyfit(relevant_pts[:, 0], relevant_pts[:, 1], 1))
                line, perc_inliers = fit_line_weighted(relevant_pts, relevant_weights, iterations=500, thresh=.5)
                # print(line)
                # print(perc_inliers)
                if np.isnan(line[0]) or np.abs(line[1]) > 1:
                    # vertical_line = np.poly1d(np.polyfit(relevant_pts[:, 1], relevant_pts[:, 0], 1))
                    vertical_line, perc_inliers = fit_line_weighted(np.fliplr(relevant_pts), relevant_weights,
                                                                    iterations=500, thresh=.5)
                    # print("vert_line",vertical_line)
                    # print(perc_inliers)
                    lines.append(np.float32([vertical_line(0), 0, vertical_line(10), 10]).reshape(1, -1))
                    if np.isnan(vertical_line[0]):
                        Log.d("Mask will be skewed, aborting")
                        return None
                else:
                    lines.append(np.float32([0, line(0), 10, line(10)]).reshape(1, -1))
            else:
                lines.append(
                    np.float32([pt1[0], pt1[1], pt2[0], pt2[1]]).reshape(1, -1))  # if no pts add original corners
        new_corners = []
        lines_array = np.concatenate(lines, axis=0)
        for idx in range(4):
            pt11 = lines_array[idx, :2]
            pt12 = lines_array[idx, 2:]
            pt21 = lines_array[(idx + 1) % 4, :2]
            pt22 = lines_array[(idx + 1) % 4, 2:]
            new_corners.append(self.intersection_two_lines_pts(pt11, pt12, pt21, pt22).reshape(1, -1))
        return np.concatenate(new_corners, axis=0)

    def fine_tune_contour_to_local_gradient_maxima(self, contour, rgb, radius=2):
        contour_only = np.zeros_like(rgb)
        rect_only = cv2.drawContours(contour_only, [contour], 0, (255, 255, 255), radius)[:, :, 0]
        rgb_gradient = self.get_laplacian(1, rgb, 3)
        pass

    def optimize_cricle_contour_RANSAC(self, table_contour, rgb):
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rgb_edges = np.uint8(self.canny_override(rgb_gray, RGB_EDGES_THRESHOLD))
        self.edges = rgb_edges  # np.uint8(self.edges*rgb_edges*255)
        cricle_drawing = np.zeros_like(rgb)
        cricle_drawing = cv2.drawContours(cricle_drawing, [table_contour], 0, (255, 255, 255), 35)[:, :, 0]

        relevant_edges = self.edges * cricle_drawing
        rgb_shape = rgb.shape
        pts = np.fliplr(np.array(np.where(relevant_edges)).T)
        # cv2.imshow("rgb_edges", np.uint8(relevant_edges>0)*255)
        # cv2.waitKey(1)

        best_ellipse = self.ransac_find_best_fit_ellipse(pts, rgb_edges, rgb_shape, num_pts_per_iter=20)

        cricle_drawing = np.zeros((rgb_shape[0], rgb_shape[1], 3), dtype=np.uint8)
        cricle_drawing = cv2.ellipse(cricle_drawing, best_ellipse, (255, 255, 255), -1)[:,:, 0]
        _, table_contour, _ = cv2.findContours(cricle_drawing.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        return np.float32(table_contour[0])  # -.5 # .5 offset due to incorrect edges location

    def ransac_find_best_fit_ellipse(self, pts, rgb_edges, rgb_shape, max_dist=1, num_pts_per_iter=100, num_iter=1000):
        corr_coeffs = []
        ellipses = []
        # for tables that are fully seen in the depth frame
        for _ in range(num_iter):
            random_pts = np.expand_dims(pts[np.random.choice(pts.shape[0], np.minimum(num_pts_per_iter, pts.shape[0]),
                                                             replace=False), :], axis=1)
            ellips_bounding_box = cv2.fitEllipse(random_pts)
            cricle_drawing = np.zeros((rgb_shape[0], rgb_shape[1], 3), dtype=np.uint8)
            cricle_drawing = cv2.ellipse(cricle_drawing, ellips_bounding_box, (255, 255, 255), max_dist)[:, :, 0]
            ellipses.append(ellips_bounding_box)
            corr_coeffs.append((cricle_drawing * rgb_edges).sum())
        best_ellipse = ellipses[np.array(corr_coeffs).argmax()]
        return best_ellipse

    def dilate_contour(self, contour, dilate_amount=5, shape='unkown'):
        if shape == 'circle':
            centroid = self.shapeDetector.get_center_of_ellipse(contour)
        else:
            centroid = self.shapeDetector.get_center_xy(contour)
        contour_relative_to_center = np.squeeze(contour, axis=1) - centroid
        dist = np.linalg.norm(contour_relative_to_center, axis=1).reshape((-1, 1))
        contour_relative_to_center = contour_relative_to_center / dist * (dist + dilate_amount)
        return np.expand_dims(contour_relative_to_center + centroid, axis=1)

    def dilate_ellipse_contour(self, contour, depth_map, dilate_amount=5):
        edges_color = cv2.drawContours(image=np.zeros_like(np.tile(depth_map[:,:,np.newaxis], (1,1,3))), color=(255,255,255),
                                 thickness=11, contours=[contour], contourIdx=0)
        edges = edges_color[:,:,0]
        ellipse = self.ransac_find_best_fit_ellipse(contour[:,0,:], edges, rgb_shape=depth_map.shape,
                                                    max_dist=dilate_amount, num_iter=1000, num_pts_per_iter=20)
        cricle_drawing = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
        cricle_drawing = cv2.ellipse(cricle_drawing, ellipse, (255, 255, 255), -1)[:, :, 0]
        _, contour, _ = cv2.findContours(cricle_drawing.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contour = np.float32(contour[0])
        edges_contour = cv2.drawContours(np.uint8(edges_color), contours=[np.int32(contour)], contourIdx=0, color=(255,0,255), thickness=1)
        # cv2.imshow("edges_contour", edges_contour)
        # cv2.waitKey(0)
        contour = self.dilate_contour(contour, dilate_amount=dilate_amount, shape="circle" )
        return contour  # -.5 # .5 offset due to incorrect edges location



    def dilate_rect_contour(self, contour, dilate_amount=5):
        centroid = self.shapeDetector.get_center_xy(contour)
        contour_relative_to_center = np.squeeze(contour, axis=1) - centroid
        # Log.d("VALUE OF contour_relative_to_center: {}".format(contour_relative_to_center))
        contour_relative_to_center = np.sign(contour_relative_to_center) * (
                np.abs(contour_relative_to_center) + dilate_amount)
        return np.expand_dims(contour_relative_to_center + centroid, axis=1)

    def round_int32(self, np_array):
        if np_array is not None:
            return np.int32(np.round(np_array))
        else:
            None

    def sort_pts_clockwise(self, pts):
        theta = self.polar_transform_theta(pts)
        return pts[np.argsort(theta), :]

    def polar_transform_theta(self, pts):
        centroid = pts.mean(axis=0)
        return np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])

    def canny_override(self, img, thresh_min=0, thresh_max=2 ** 16, kernel_size=3, factor=1):
        laplacian = self.get_laplacian(factor, img, kernel_size)
        return (laplacian >= thresh_min) * (laplacian <= thresh_max)

    def get_laplacian(self, factor, img, kernel_size):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size) / factor
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size) / factor
        laplacian = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
        return laplacian

    def distance_two_pts_line_to_pts(self, pt1, pt2, pts):
        return np.abs((pt2[1] - pt1[1]) * pts[:, 0] - (pt2[0] - pt1[0]) * pts[:, 1]
                      + pt2[0] * pt1[1] - pt2[1] * pt1[0]) / \
               np.sqrt((pt2[1] - pt1[1]) ** 2 + (pt2[0] - pt1[0]) ** 2)

    def intersection_two_lines_pts(self, pt11, pt12, pt21, pt22):
        return np.array([((pt11[0] * pt12[1] - pt11[1] * pt12[0]) * (pt21[0] - pt22[0]) - (pt11[0] - pt12[0]) * (
                pt21[0] * pt22[1] - pt21[1] * pt22[0])) /
                         (((pt11[0] - pt12[0]) * (pt21[1] - pt22[1])) - (pt11[1] - pt12[1]) * (pt21[0] - pt22[0])),

                         ((pt11[0] * pt12[1] - pt11[1] * pt12[0]) * (pt21[1] - pt22[1]) - (pt11[1] - pt12[1]) * (
                                 pt21[0] * pt22[1] - pt21[1] * pt22[0])) /
                         (((pt11[0] - pt12[0]) * (pt21[1] - pt22[1])) - (pt11[1] - pt12[1]) * (pt21[0] - pt22[0]))])

    def display(self, depth_map, contours, edges=None):
        # dmap_disp = cv2.cvtColor(np.uint8(depth_map*.05), cv2.COLOR_GRAY2BGR)
        dmap_disp = self.draw_table_on_rbg(depth_map, contours)
        FileUploader.save_image(dmap_disp, "table_detector_{}.png".format(self.image_index), folder="table_detection/")
        # cv2.imshow('detected shapes', cv2.resize(dmap_disp, (0, 0), fx=.5, fy=.5))
        # cv2.waitKey(1)
        if edges is not None:
            FileUploader.save_image(edges, "table_detector_edges_{}.png".format(self.image_index), folder="table_detection/")
            # cv2.imshow('edges', edges)
            # cv2.waitKey(1)
        self.image_index += 1

    def draw_table_on_rbg(self, rbg, contours):
        if len(rbg.shape) < 3:
            dmap_disp = cv2.cvtColor(np.uint8(rbg), cv2.COLOR_GRAY2BGR)
        else:
            dmap_disp = rbg
        # Log.d("Contours: {}".format(len(contours)))
        for contour in contours:
            # Log.d("Contour: {}".format(contour))
            if contour is not None:
                shape = self.shapeDetector.detect(contour)
                # Log.d("Shape: {}".format(shape))
                if shape != 'unidentified':
                    # Log.d("Drawing unidentified")
                    dmap_disp = cv2.drawContours(dmap_disp, [contour], 0, (255, 0, 255), 1)
                if shape == 'rectangle':
                    # Log.d("Drawing rectangle")
                    dmap_disp = cv2.drawContours(dmap_disp, [contour], 0, (255, 0, 0), 1)
                if shape == 'circle':
                    # Log.d("Drawing circle")
                    dmap_disp = cv2.drawContours(dmap_disp, [contour], -1, (0, 255, 0), 1)
                # else:
                #     dmap_disp = cv2.drawContours(dmap_disp, contour, -1, (255, 0, 255), 2)
        return dmap_disp


def get_warp_320x240_640x480(rgb_large, rgb_small):
    rgb_small = cv2.resize(rgb_small, (rgb_large.shape[1], rgb_large.shape[0]))
    rgb_mean_reduced = (rgb_large.mean(axis=2))  # - rgb_large.mean(axis=2).mean()
    rgb_small = (rgb_small.mean(axis=2))  # - rgb_small.mean(axis=2).mean()

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-12)
    (cc, tform_to_big_to_small) = cv2.findTransformECC(np.float32(rgb_mean_reduced), np.float32(rgb_small),
                                                       np.eye(3, 3, dtype=np.float32),
                                                       motionType=cv2.MOTION_HOMOGRAPHY, criteria=criteria)
    return tform_to_big_to_small

    # def create_dist_mask(self, radius):
    #     disk = np.zeros((radius*2+1, radius*2+1))
    #     dist_from_center =
    #
    #
    # def create_white_edgeless_background(self):

# if __name__ == '__main__':
#
#     FrameManager.getInstance().start()
#
#     tableDetector = TableDetector()
#
#     cv2.namedWindow("white_background", cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty("white_background", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#     white_screen = np.float32(np.zeros((240, 320, 3)))
#     white_screen[5:-5, 5:-5, :] = 1
#     white_screen = np.uint8(cv2.filter2D(white_screen, -1, np.ones((21,21))/21/21*255))
#     cv2.imshow("white_background", white_screen)
#     cv2.waitKey(1)
#
#     keep_running = True
#     while keep_running:
#         key = cv2.waitKey(1)
#         if key == 60:
#             keep_running = False
#
#         # dmap = .05 * cam.get_depth()
#         # rgb = .05 * cam.get_rgb()
#         #
#         # for _ in range(19):
#         #     dmap += .05 * cam.get_depth()
#         #     rgb += .05 * cam.get_rgb()
#
#         FrameManager.getInstance().set_depth_resolution(Globals.DEPTH_HIGH_QUALITY)
#         FrameManager.getInstance().set_rgb_resolution(Globals.RGB_MEDIUM_QUALITY)
#         depth_frame = FrameManager.getInstance().get_depth_frame()
#         cv2.waitKey(10)
#         dmap = cv2.resize(depth_frame, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
#         rbg_small = FrameManager.getInstance().get_rgb_frame()
#         cv2.waitKey(10)
#
#         FrameManager.getInstance().set_rgb_resolution(Globals.RGB_HIGH_QUALITY)
#         cv2.waitKey(10)
#         rgb = FrameManager.getInstance().get_rgb_frame()
#         cv2.waitKey(10)
#
#         tform_small_to_big = get_warp_320x240_640x480(rgb, rbg_small)
#         dmap = cv2.warpPerspective(dmap, np.linalg.inv(tform_small_to_big),
#                                    (dmap.shape[1], dmap.shape[0]), cv2.INTER_NEAREST)
#
#         # dmap = cam.get_depth()
#         # rgb = cam.get_rgb()
#         tableDetector.detect_table(dmap, rgb, display=True, save_path="./table_contour_test.png")
#
#     ## Release resources
#     cv2.destroyAllWindows()
