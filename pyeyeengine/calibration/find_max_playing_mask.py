import cv2
import numpy as np
import os

from pyeyeengine.object_detection.shape_detector import ShapeDetector

shapeDetector = ShapeDetector()


def find_max_playing_mask(calibrator, is_display=False):
    mask_for_cam = calibrator.table_mask
    if len(mask_for_cam.shape) == 3:
        mask_for_cam = mask_for_cam[:, :, 0]
    # table_contour_cam = calibrator.table_contour
    mask_for_display = cv2.warpPerspective(mask_for_cam, calibrator.warp_mat_cam_2_displayed,
                                           dsize=(calibrator.screen_width, calibrator.screen_height))
    # _, surface_contour_display, _ = cv2.findContours(mask_for_display, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if calibrator.table_contour is None:
        return mask_for_display

    surface_contour_display = cv2.transform(np.float32(calibrator.table_contour), calibrator.warp_mat_cam_2_displayed)
    surface_contour_display = surface_contour_display[:, 0, :2].reshape(-1, 1, 2) / \
                              surface_contour_display[:, 0, 2].reshape((-1, 1, 1))
    # surface_contour_display[:, 0, 1] -= 80

    surface_shape = calibrator.table_shape
    playing_mask = np.tile(np.expand_dims(np.zeros_like(mask_for_display), axis=2), (1, 1, 3))

    if surface_shape == 'rectangle':
        playing_mask = cv2.drawContours(image=np.zeros_like(playing_mask),
                                        contours=[np.int32(np.round(surface_contour_display))],
                                        contourIdx=0,
                                        color=(255, 255, 255),
                                        thickness=-1)[:, :, 0]
    elif surface_shape == 'circle':
        center, radius, _, _ = shapeDetector.fit_ellipse_to_contour(surface_contour_display)
        ellips_bounding_box = cv2.fitEllipse(np.float32(surface_contour_display))
        playing_mask = cv2.ellipse(playing_mask, ellips_bounding_box, (255, 255, 255), -1)[:, :, 0]
    else:
        playing_mask = np.ones_like(mask_for_display) * 255

    BASE_PATH = os.path.dirname(os.path.realpath(__file__))
    cv2.imwrite(BASE_PATH + "/playing_mask.png", playing_mask)
    if is_display:
        playing_mask = display(playing_mask, surface_contour_display)
    return playing_mask


def display(playing_mask, surface_contour_display):
    cv2.namedWindow("mask_for_display", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("mask_for_display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # playing_mask = np.uint8(cv2.blur(playing_mask, (25,25)))
    # playing_mask = np.uint8(playing_mask)
    playing_mask = np.tile(np.expand_dims(playing_mask, axis=2), (1, 1, 3))

    playing_mask = cv2.drawContours(image=playing_mask, contours=[np.int32(np.round(surface_contour_display))],
                                    contourIdx=0,
                                    color=(255, 0, 255),
                                    thickness=1)
    cv2.imshow('mask_for_display', playing_mask)
    cv2.waitKey(30)
    # from pyeyeengine.camera_utils.rgb_camera_reader import RGBCameraReader
    # camera = RGBCameraReader(display=False, resxy=(1280, 960))
    # rgb_after = camera.get_rgb()
    # cv2.imwrite( "./table_mask_from_above.png",rgb_after)
    # camera.stop()
    # cv2.imshow('mask_for_display', playing_mask)
    # cv2.waitKey(0)
    return playing_mask


def limit_contour_to_screen(contour, screen_width, screen_height):
    contour = np.maximum(contour, 0)
    contour[contour[:, 0, 0] > screen_width, 0, 0] = screen_width
    contour[contour[:, 0, 1] > screen_height, 0, 1] = screen_height
    return contour


def fit_circle_smart(pts):
    a = np.linalg.lstsq(np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1),
                        (-1 * (pts ** 2).sum(axis=1)).reshape((-1, 1)))[0]
    xc = -.5 * a[0]
    yc = -.5 * a[1]
    R = np.sqrt((a[:2] ** 2).sum() / 4 - a[2])
    # a = [x y ones(size(x))]\[-(x. ^ 2 + y. ^ 2)];
    # xc = -.5 * a(1);
    # yc = -.5 * a(2);
    # R = sqrt((a(1) ^ 2 + a(2) ^ 2) / 4 - a(3));
    return np.array([xc, yc]), R


def calc_2_longer_sides_of_rect(corners):
    p1 = corners[0, :]
    p2_options = corners[1:, :]
    ind = np.argsort(dist_pt_pts(p1, p2_options))
    p2 = p2_options[ind[1], :]
    return two_pts_to_line(p1, p2), two_pts_to_line(p2_options[ind[0], :], p2_options[ind[-1], :])


def corners_to_max_upright_playing_region(corners):
    xrange = np.sort(corners[:, 0])[1:-1]
    yrange = np.sort(corners[:, 1])[1:-1]
    return np.array([[i, j] for i in yrange.tolist() for j in xrange.tolist()])


def get_maximal_aspect_ratio_screen_region(corners, aspect_ratios):
    playable_region_corners = corners_to_max_upright_playing_region(corners)
    centroid = playable_region_corners.mean(axis=0)
    max_width = playable_region_corners[:, 1].max() - playable_region_corners[:, 1].min()
    max_height = playable_region_corners[:, 0].max() - playable_region_corners[:, 0].min()

    rects = []
    areas = []
    for aspect_ratio in aspect_ratios:
        width_is_limiting_factor = max_width / max_height < aspect_ratio[0] / aspect_ratio[1]
        if width_is_limiting_factor:
            max_height = aspect_ratio[1] / aspect_ratio[0] * max_width
        else:
            max_width = aspect_ratio[0] / aspect_ratio[1] * max_height
        rects.append(
            [int(np.round(centroid[1] - max_width / 2)), int(np.round(centroid[0] - max_height / 2)), max_width,
             max_height])
        areas.append(max_height * max_width)

    return rects[np.array(areas).argmax()]


def find_optimal_playing_screen(table_contour, screen_width, screen_height):
    rect = cv2.minAreaRect(np.int32(np.round(table_contour)))
    corners = np.int0(cv2.boxPoints(rect))
    corners = np.maximum(corners, 0)
    corners[corners[:, 0] > screen_width, 0] = screen_width
    corners[corners[:, 1] > screen_height, 1] = screen_height
    centroid_xy = corners.mean(axis=0)

    bounding_boxes = []
    areas = []
    aspect_ratios = [np.array([1, 1]), np.array([4, 3]), np.array([16, 9])]

    for aspect_ratio_xy in aspect_ratios:
        play_screen_corner_1, play_screen_corner_2 = find_opposite_corners_of_play_screen(corners, centroid_xy,
                                                                                          aspect_ratio_xy)
        bounding_box = opposite_corners_to_bounding_box(np.array(play_screen_corner_1).reshape(1, 2),
                                                        np.array(play_screen_corner_2).reshape(1, 2))
        bounding_boxes.append(bounding_box)
        areas.append(bounding_box[-1] * bounding_box[-2])

    return bounding_boxes[np.array(areas).argmax()]


def opposite_corners_to_bounding_box(play_screen_corner_1, play_screen_corner_2):
    corners = np.concatenate([play_screen_corner_1, play_screen_corner_2], axis=0)
    minxy = corners.min(axis=0)
    maxxy = corners.max(axis=0)
    return [int(np.ceil(minxy[0])), int(np.ceil(minxy[1])), int(np.floor(maxxy[0] - minxy[0])),
            int(np.floor(maxxy[1] - minxy[1]))]


def get_cross_section_lines(center_of_surface, aspect_ratio_xy):
    line1 = [aspect_ratio_xy[1] / aspect_ratio_xy[0],
             (center_of_surface[1] - aspect_ratio_xy[1] / aspect_ratio_xy[0] * center_of_surface[0])]
    line2 = [-aspect_ratio_xy[1] / aspect_ratio_xy[0],
             (center_of_surface[1] + aspect_ratio_xy[1] / aspect_ratio_xy[0] * center_of_surface[0])]
    return line1, line2  # format : [m, b] -> y = mx + b


def find_opposite_corners_of_play_screen(corners, center, aspect_ratio_xy):
    side_line_1, side_line_2 = calc_2_longer_sides_of_rect(corners)
    cross_section_line_1, cross_section_line_2 = get_cross_section_lines(center, aspect_ratio_xy)

    play_screen_corner_1_1 = line_intersect(side_line_1, cross_section_line_1)
    play_screen_corner_1_2 = line_intersect(side_line_1, cross_section_line_2)

    if dist_pt_pts(center, play_screen_corner_1_1) < dist_pt_pts(center, play_screen_corner_1_2):
        play_screen_corner_1 = play_screen_corner_1_1
        play_screen_corner_2 = line_intersect(side_line_2, cross_section_line_1)
    else:
        play_screen_corner_1 = play_screen_corner_1_2
        play_screen_corner_2 = line_intersect(side_line_2, cross_section_line_2)

    return play_screen_corner_1, play_screen_corner_2


def dist_pt_pts(pt, pts):
    return np.linalg.norm(pt - pts, axis=1)


def two_pts_to_line(pt1, pt2):
    m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    b = pt1[1] - m * pt1[0]
    return [m, b]


def line_intersect(line1, line2):
    m1, b1 = line1  # slope, y-intersect
    m2, b2 = line2
    if m1 == m2:  # parallel line
        return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return np.array([x, y]).reshape((1, 2))
