import cv2
import numpy as np
from pyeyeengine.utilities.logging import Log

# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
class ShapeDetector:
    def __init__(self):
        pass

    def is_contour_rectangle(self, contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        best_fit_rotated_rectangle = np.int0(box)
        area_best_fit = cv2.contourArea(best_fit_rotated_rectangle)
        area_actual = cv2.contourArea(contour)
        return (area_best_fit > area_actual) and (area_actual / area_best_fit > .85)

    def is_contour_circle(self, contour):
        # turn to normal points.
        xy = np.squeeze(np.asarray(contour), axis=1)
        # find centroid.
        contour_center = self.get_center_xy(contour)
        # find dist to each point
        r, theta = self.polar(xy[:, 0] - contour_center[0], xy[:, 1] - contour_center[1])
        # Log.d("VALUE OF r: {}".format(r))
        return np.std(r) / r.mean() < .1  # if cv is low

    def fit_ellipse_to_contour(self, contour):
        ellips_bounding_box = cv2.fitEllipse(np.float32(contour))
        center = np.array(ellips_bounding_box[0]).reshape((1, -1))
        r = np.array(ellips_bounding_box[1]) / 2
        rotation_angle = np.array(ellips_bounding_box[2])
        return center, r.min(), r.max(), rotation_angle

    def polar(self, x, y):
        """returns r, theta(degrees)"""
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return r, theta

    def get_center_xy(self, contour):
        M = cv2.moments(contour)
        c_x = int(M["m10"] / max([M["m00"], 1]))
        c_y = int(M["m01"] / max([M["m00"], 1]))
        return np.array([c_x, c_y])

    def find_dist_from_closest_pt_in_contour_to_a_pt(self, contour, x, y):
        polygon = cv2.drawContours(
            np.zeros((max(contour[:, 0, 1].max(), y) + 1, max(contour[:, 0, 0].max(), x) + 1, 3), dtype=np.uint8),
            [contour], 0, thickness=-1, color=(255, 255, 255))
        if polygon[y, x, 0] > 0:  # cv2.pointPolygonTest(contour, (x,y), False):
            return 0
        else:
            pt = np.array([x, y]).reshape(1, 2)
            return np.sqrt(np.power(np.squeeze(contour, axis=1) - pt, 2).sum(axis=1)).min()

    def get_center_of_circle(self, contour):
        pts = np.squeeze(contour, axis=1)
        a = np.linalg.lstsq(np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1),
                            (-1 * (pts ** 2).sum(axis=1)).reshape((-1, 1)))[0]
        xc = -.5 * a[0]
        yc = -.5 * a[1]
        # R = np.sqrt((a[:2] ** 2).sum() / 4 - a[2])
        return np.array([np.round(xc), np.round(yc)]).reshape((1, -1))

    def get_center_of_ellipse(self, contour):
        ellips_bounding_box = cv2.fitEllipse(contour)
        return np.array(ellips_bounding_box[0]).reshape((1, -1))

    def detect(self, contour):
        shape = "unidentified"  # initialize the shape
        if contour is None:
            return shape

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.15 * peri, True)

        if len(approx) == 4 or self.is_contour_rectangle(contour):
            shape = "rectangle"

        if (self.is_contour_circle(contour)):
            shape = "circle"

        Log.i("[TABLE] Found table shape", extra_details={"shape": shape})

        # otherwise, we don't know
        return shape
