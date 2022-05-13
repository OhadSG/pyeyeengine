import cv2
import numpy as np

from pyeyeengine.camera_utils.camera_manager import CameraManager
from pyeyeengine.camera_utils.camera_reader import CameraReader
from pyeyeengine.object_detection.table_detector import TableDetector


def try_to_find_table(_camera):
    for try_num in range(10):
        depth_map = cv2.resize(_camera.get_depth(res_xy=(640, 480)), (0, 0), fx=2, fy=2,
                               interpolation=cv2.INTER_NEAREST)
        rbg = _camera.get_rgb()
        rgb_big = cv2.resize(rbg, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        rgb_enlarged = _camera.get_rgb(res_xy=(1280, 960))

        tform_small_to_big = get_warp_ORB(rgb_enlarged, rbg)
        # if not check_tform_reasonable(tform_small_to_big):
        #     tform_small_to_big = np.eye(3)
        # tform_small_to_big = get_warp_320x240_640x480(rgb_enlarged, rbg)
        depth_map_tformed = cv2.warpPerspective(depth_map, np.linalg.inv(tform_small_to_big),
                                                (depth_map.shape[1], depth_map.shape[0]), cv2.INTER_NEAREST)

        rgb_tformed = cv2.warpPerspective(rgb_big,
                                          np.linalg.inv(tform_small_to_big),
                                          (depth_map.shape[1], depth_map.shape[0]), cv2.INTER_NEAREST)

        return rgb_enlarged, rgb_tformed, depth_map_tformed


def rgb_res_offset(rgb_large, rgb_small, table_contour, radius=4):
    rgb_small = cv2.resize(rgb_small, (rgb_large.shape[1], rgb_large.shape[0]))
    rgb_mean_reduced = (rgb_large.mean(axis=2)) - rgb_large.mean(axis=2).mean()
    # rgb_mean_reduced = cv2.resize(rgb_mean_reduced, (rgb_small.shape[1], rgb_small.shape[0]))
    rgb_small = (rgb_small.mean(axis=2)) - rgb_small.mean(axis=2).mean()
    rgb_small_cropped = rgb_small[radius:-radius, radius:-radius]

    corr = cv2.matchTemplate(np.float32(rgb_mean_reduced), np.float32(rgb_small_cropped), cv2.TM_CCORR_NORMED)
    offset = np.array(np.where(corr == corr.max())) - radius
    return np.fliplr(offset.reshape(1, -1))  # xy


def get_warp_320x240_640x480(rgb_large, rgb_small, mask=None):
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


def get_warp_ORB(rgb_large, rgb_small, mask=None):
    rgb_small = cv2.resize(rgb_small, (rgb_large.shape[1], rgb_large.shape[0]))
    if mask is None:
        mask = np.ones((rgb_large.shape[0], rgb_large.shape[1]))
    rgb_mean_reduced = (rgb_large.mean(axis=2)) * np.uint8(mask > 0)  # - rgb_large.mean(axis=2).mean()
    rgb_small = (rgb_small.mean(axis=2)) * np.uint8(mask > 0)  # - rgb_small.mean(axis=2).mean()

    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = float(open("./good_match_perc.txt", "r").read())  # 0.15
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
    return h


def check_tform_reasonable(tform_small_to_big):
    return (np.abs(tform_small_to_big[0, 0] - 1) < .05) and (np.abs(tform_small_to_big[1, 1] - 1) < .05) and \
           (np.abs(tform_small_to_big[2, 2] - 1) < .05) and (np.abs(tform_small_to_big[0, 1]) < .05) and \
           (np.abs(tform_small_to_big[1, 0]) < .05) and np.all(np.abs(tform_small_to_big[2, :2]) < .05)


if __name__ == '__main__':
    cv2.namedWindow("white_background", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("white_background", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    white_screen = np.float32(np.zeros((240, 320, 3)))
    white_screen[5:-5, 5:-5, :] = 1
    white_screen = np.uint8(cv2.filter2D(white_screen, -1, np.ones((21, 21)) / 21 / 21 * 255))
    cv2.imshow("white_background", white_screen)
    cv2.waitKey(1)

    _camera = CameraManager()
    while True:
        rgb_big, rgb_tformed, depth_map_tformed = try_to_find_table(_camera)

        rgb_overlap = rgb_big
        rgb_overlap[:, :, 1] = rgb_tformed[:, :, 1]
        rgb_overlap = cv2.resize(rgb_overlap, (320, 240))

        cv2.imshow("rgb_rough_contour", rgb_overlap)
        cv2.waitKey(1)
