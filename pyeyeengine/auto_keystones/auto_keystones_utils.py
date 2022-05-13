import time

import cv2
import numpy as np

from pyeyeengine.eye_engine.ransac_utils import fit_plane
from pyeyeengine.projector_controller.projector_controller import ProjectorController


def find_trapazoid_size_angles(bottom_left, bottom_right, top_left, top_right):
    bottom_left, bottom_right = bottom_left.reshape(-1), bottom_right.reshape(-1)
    top_left, top_right = top_left.reshape(-1), top_right.reshape(-1)
    thetha_left = angle_about_vertex(bottom_left, bottom_right, top_left)
    thetha_right = angle_about_vertex(bottom_right, top_right, bottom_left)
    print("thetha_left: ", thetha_left, ", thetha_right: ", thetha_right)
    return (thetha_left + thetha_right) / 2


def angle_about_vertex(pt_vertex, pt2, pt3):
    # https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
    return inner_angle(np.rad2deg(np.arctan2(pt2[1] - pt_vertex[1], pt2[0] - pt_vertex[0]) -
                                  np.arctan2(pt3[1] - pt_vertex[1], pt3[0] - pt_vertex[0])))


def inner_angle(deg):
    return np.mod(deg, 180) - 90


def angle_between_two_lines(line_1, line_2):
    # https://www.wikihow.com/Find-the-Angle-Between-Two-Vectors
    dot_product_lines = np.sum(line_1 * line_2)
    inner_product_line_1 = np.sqrt(np.sum(line_1 * line_1))
    inner_product_line_2 = np.sqrt(np.sum(line_2 * line_2))
    cos_theta = dot_product_lines / (inner_product_line_1 * inner_product_line_2)
    return np.rad2deg(np.arccos(cos_theta))


def find_vertical_keystone_angle(bottom_left, bottom_right, top_left, top_right):
    line_left = top_left - bottom_left
    line_right = top_right - bottom_right
    angle = angle_between_two_lines(line_left, line_right) / 2
    if np.linalg.norm(bottom_left - bottom_right) > np.linalg.norm(top_left - top_right):
        angle *= -1
    return angle


def find_horizontal_keystone_angle(bottom_left, bottom_right, top_left, top_right):
    line_top = top_left - top_right
    line_bottom = bottom_left - bottom_right
    angle = angle_between_two_lines(line_top, line_bottom) / 2
    if np.linalg.norm(top_right - bottom_right) > np.linalg.norm(top_left - bottom_left):
        angle *= -1
    return angle


def find_rectification_tform(depth_map):
    x, y, z = depth_to_xyz(depth_map)

    # plot_depth_3d(x, y, z)

    voxels = xyz_to_valid_voxels(x, y, z)
    plane_tform = fit_plane(voxels, iterations=20, inlier_thresh=25, num_pts_per_param=1000, perc_validation=1)
    corners_xy1 = np.array([[0, 0, 1], [0, depth_map.shape[0], 1], [depth_map.shape[1], 0, 1],
                            [depth_map.shape[1], depth_map.shape[0], 1]])
    corners_z = np.matmul(corners_xy1, plane_tform)
    center_height = depth_map[int(depth_map.shape[0] / 2), int(depth_map.shape[1] / 2)]  # 2000
    corners_z_proportion = corners_z / center_height  # maybe backwards
    dist_to_dorners = corners_xy1[:, :-1] - np.array([[int(depth_map.shape[0] / 2), int(depth_map.shape[1] / 2)]])
    corners_xy_transformed = (dist_to_dorners * corners_z_proportion) + np.array(
        [[int(depth_map.shape[0] / 2), int(depth_map.shape[1] / 2)]])

    rectification_tform, status = cv2.findHomography(np.expand_dims(corners_xy1[:, :-1], axis=1),
                                                     np.expand_dims(corners_xy_transformed, axis=1))  # src, dst
    return rectification_tform


def clean_pairs(rect_x, rect_y, rgb_x, rgb_y, rgb):
    xys = np.hstack([rect_x.reshape(-1, 1), rect_y.reshape(-1, 1), rgb_x.reshape(-1, 1), rgb_y.reshape(-1, 1)])
    xys = round_int(xys)
    xys = xys[((xys[:, 0] > 0) * (xys[:, 0] < rgb.shape[1])) > 0, :]  # keep only x values in rgb's range
    xys = xys[((xys[:, 1] > 0) * (xys[:, 1] < rgb.shape[0])) > 0, :]  # keep only y values in rgb's range
    return np.hsplit(xys, 4)


def round_int(np_array):
    return np.int32(np.round(np_array))


def xyz_to_valid_voxels(x, y, z):
    voxels = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])
    voxels = voxels[voxels[:, -1] > 0, :]
    return voxels


def depth_to_xyz(depth_map):
    x, y = np.meshgrid(np.linspace(0, depth_map.shape[1] - 1, depth_map.shape[1]),
                       np.linspace(0, depth_map.shape[0] - 1, depth_map.shape[0]))
    z = depth_map
    return x, y, z


def plot_depth_3d(x, y, z):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(np.int32(x[::20, ::20]), np.int32(y[::20, ::20]),
               z[np.int32(y[::20, ::20]), np.int32(x[::20, ::20])])
    plt.show()


def fix_keystones(calibrator):
    # ProjectorController().change_vertical_keystone(0)
    bottom_left, bottom_right, top_left, top_right = calibrator.get_display_corners_on_cam()

    theta = find_trapazoid_size_angles(bottom_left, bottom_right, top_left, top_right)
    # ProjectorController().change_vertical_keystone(theta)

    bottom_left, bottom_right = bottom_left.reshape(-1), bottom_right.reshape(-1)
    top_left, top_right = top_left.reshape(-1), top_right.reshape(-1)
    theta_vertical = find_vertical_keystone_angle(bottom_left, bottom_right, top_left, top_right)
    theta_horizontal = find_horizontal_keystone_angle(bottom_left, bottom_right, top_left, top_right)
    ProjectorController().change_vertical_keystone(theta_vertical)
    ProjectorController().change_horizontal_keystone(theta_horizontal)

    time.sleep(2)
    print("theta_vertical : %f" % theta_vertical)
    print("theta_horizontal : %f" % theta_horizontal)


def fix_keystones_using_epsons_autokeystones():
    for _ in range(10):
        original_view_state_ceiling = ProjectorController().get_view_ceiling()
        if original_view_state_ceiling == "ON" or original_view_state_ceiling == "OFF":
            break

    for _ in range(10):
        original_view_state_rear = ProjectorController().get_view_rear()
        if original_view_state_rear == "ON" or original_view_state_rear == "OFF":
            break

    print("is_ciel: %s, is_rear: %s" % (original_view_state_ceiling, original_view_state_rear))
    if original_view_state_ceiling == "ON":
        ProjectorController().set_view_ceiling("OFF")
    if original_view_state_rear == "ON":
        ProjectorController().set_view_rear("OFF")

    ProjectorController().set_auto_keystones("ON")
    time.sleep(3)

    vertical_keystones = int(ProjectorController().epscom_get("VKEYSTONE?"))
    time.sleep(2)
    horizonal_keystones = int(ProjectorController().epscom_get("HKEYSTONE?"))
    time.sleep(2)
    print("vkey: %s, hkey: %s" % (vertical_keystones, horizonal_keystones))

    if original_view_state_rear == "ON":  # floor
        ProjectorController().set_view_rear("ON")
        if original_view_state_ceiling == "ON":
            ProjectorController().set_view_ceiling("ON")
            ProjectorController().epscom_set(
                "VKEYSTONE %d" % (255 - max(vertical_keystones - 25, 0)))  # -40 ~= 5 degrees
            print("vkey: %s" % (255 - max(vertical_keystones - 25, 0)))
        else:
            ProjectorController().set_view_ceiling("OFF")
            ProjectorController().epscom_set("VKEYSTONE %d" % max(vertical_keystones - 25, 0))  # -40 ~= 5 degrees
            print("vkey: %s" % max(vertical_keystones - 25, 0))
        ProjectorController().epscom_set("HKEYSTONE %d" % (horizonal_keystones))
        print("hkey: %s" % (horizonal_keystones))


    elif original_view_state_ceiling == "ON":  # wall
        ProjectorController().set_view_ceiling("ON")
        ProjectorController().set_view_rear("OFF")
        ProjectorController().epscom_set("HKEYSTONE %d" % (int(horizonal_keystones)))
        print("hkey: %s" % (horizonal_keystones))
        ProjectorController().epscom_set("VKEYSTONE %d" % int(vertical_keystones-7))
        print("vkey: %s" % int(vertical_keystones-7))
    else:
        raise Exception("unsupported projector mode ( front ) ")

    # if original_view_state_ceiling == "ON": # wall
    #     ProjectorController().set_view_ceiling("ON")
    #     ProjectorController().epscom_set("VKEYSTONE %d" % vertical_keystones)
    #     print("vkey: %s" %vertical_keystones)
    #     if original_view_state_rear == "ON":
    #         ProjectorController().set_view_rear("ON")
    #         ProjectorController().epscom_set("HKEYSTONE %d" % (255 - horizonal_keystones))
    #         print("hkey: %s" % (255 - horizonal_keystones))
    #     else:
    #         ProjectorController().epscom_set("HKEYSTONE %d" % (horizonal_keystones))
    #         print("hkey: %s" % (horizonal_keystones))
    #
    # else: # projector not on cieling mode = floor
    #     ProjectorController().epscom_set("HKEYSTONE %d" % (horizonal_keystones))
    #     print("hkey: %s" % (horizonal_keystones))
    #
    #     if original_view_state_rear == "ON":
    #         ProjectorController().set_view_rear("ON")
    #         ProjectorController().epscom_set("VKEYSTONE %d" % max(vertical_keystones - 40, 0)) # -40 ~= 5 degrees
    #         print("vkey: %s" % max(vertical_keystones - 40, 0))
    #     else:
    #         ProjectorController().epscom_set("VKEYSTONE %d" % min(vertical_keystones + 18, 255)) #+18 ~= 5 degrees
    #         print("vkey: %s" % (min(vertical_keystones + 18, 255)))

    print("reached end of auto keystones")


if __name__ == '__main__':
    fix_keystones()
