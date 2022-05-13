import cv2
import numpy as np
import os
import configparser

from pyeyeengine.calibration.find_max_playing_mask import find_max_playing_mask
from pyeyeengine.object_detection.shape_detector import ShapeDetector

shapeDetector = ShapeDetector()

applications_ini_path = r'C:\Program Files (x86)\EyeClick\Applications\Global\applications.ini'
SceneMaskManual_bmp_path = r"C:\Program Files (x86)\EyeClick\Server\SceneMaskManual.bmp"
SceneMaskManual_png_path = r"C:\Program Files (x86)\EyeClick\Server\SceneMaskManual.png"
eyeclick_ini_path = r'C:\Program Files (x86)\EyeClick\Server\EyeClick.ini'
eyestep_ini_path = r'C:\Program Files (x86)\EyeClick\Server\EyeStep.ini'


def export_data_for_games_and_cpp_engine(calibrator):
    # mask_for_cam = calibrator.table_mask
    # table_contour_cam = calibrator.table_contour
    mask_for_display = playing_mask = find_max_playing_mask(calibrator)

    contour_as_pts = get_contours_as_points(mask_for_display)
    rect_display = np.concatenate([contour_as_pts.min(axis=0), contour_as_pts.max(axis=0)], axis=0)  # minx miny width height
    xOffset, yOffset = contour_as_pts.mean(axis=0)

    surface_shape = calibrator.table_shape

    export_to_applications_ini(calibrator, rect_display, surface_shape, xOffset, yOffset)
    export_to_eyeclick_ini(calibrator, xOffset, yOffset)
    export_to_eyestep_ini(calibrator)
    export_playing_mask_img(playing_mask)

def export_data_for_android(calibrator):
    mask_for_display = find_max_playing_mask(calibrator)

    contour_as_pts = get_contours_as_points(mask_for_display)
    rect_display = np.concatenate([contour_as_pts.min(axis=0), contour_as_pts.max(axis=0)],
                                  axis=0)  # minx miny width height
    return mask_for_display, rect_display

def get_contours_as_points(mask_for_display):
    _, surface_contour_display, _ = cv2.findContours(mask_for_display.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contour_as_pts = np.squeeze(surface_contour_display[0], axis=1)
    return contour_as_pts


def export_playing_mask_img(playing_mask):
    mask_to_save_to_img = cv2.cvtColor(np.uint8((playing_mask.copy() > 0) * 255), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(SceneMaskManual_bmp_path, mask_to_save_to_img)
    mask_as_alpha = np.concatenate([np.zeros_like(mask_to_save_to_img),
                                    np.expand_dims(255 - mask_to_save_to_img[:, :, 0], axis=2)], axis=2)
    cv2.imwrite(SceneMaskManual_png_path, mask_as_alpha)


def export_to_applications_ini(calibrator, rect_display, surface_shape, xOffset, yOffset):
    # Applications.ini
    applications_ini = get_ini_as_list(applications_ini_path)
    '''pAR0 - bottom left
                pAR1 - top left
                pAR2 - top right
                pAR3 - bottom right
                00 at top left '''
    applications_ini['Application']['p0x'] = "%d" % int(rect_display[0])
    applications_ini['Application']['p0y'] = "%d" % int(rect_display[3])
    applications_ini['Application']['p1x'] = "%d" % int(rect_display[0])
    applications_ini['Application']['p1y'] = "%d" % int(rect_display[1])
    applications_ini['Application']['p2x'] = "%d" % int(rect_display[2])
    applications_ini['Application']['p2y'] = "%d" % int(rect_display[1])
    applications_ini['Application']['p3x'] = "%d" % int(rect_display[2])
    applications_ini['Application']['p3y'] = "%d" % int(rect_display[3])
    applications_ini['Application']['maskWidth'] = "%f" % ((rect_display[2] - rect_display[0]))
    applications_ini['Application']['maskHeight'] = "%f" % ((rect_display[3] - rect_display[1]))
    applications_ini['Application']['pAR0x'] = "%f" % (rect_display[0] / calibrator.screen_width)
    applications_ini['Application']['pAR0y'] = "%f" % (rect_display[3] / calibrator.screen_height)
    applications_ini['Application']['pAR1x'] = "%f" % (rect_display[0] / calibrator.screen_width)
    applications_ini['Application']['pAR1y'] = "%f" % (rect_display[1] / calibrator.screen_height)
    applications_ini['Application']['pAR2x'] = "%f" % (rect_display[2] / calibrator.screen_width)
    applications_ini['Application']['pAR2y'] = "%f" % (rect_display[1] / calibrator.screen_height)
    applications_ini['Application']['pAR3x'] = "%f" % (rect_display[2] / calibrator.screen_width)
    applications_ini['Application']['pAR3y'] = "%f" % (rect_display[3] / calibrator.screen_height)
    applications_ini['Application']['maskWidthAR'] = "%f" % ((rect_display[2] - rect_display[0]) / calibrator.screen_width)
    applications_ini['Application']['maskHeightAR'] = "%f" % ((rect_display[3] - rect_display[1]) / calibrator.screen_height)
    applications_ini['Application']['Height'] = "%d" % (calibrator.screen_height)
    applications_ini['Application']['Width'] = "%d" % (calibrator.screen_width)
    applications_ini['ProjOffset']['shape'] = 'false'  # isRound
    if surface_shape == 'rectangle':
        ratio = (rect_display[2] - rect_display[0]) / (rect_display[3] - rect_display[1])
        if ratio >= 16 / 9:
            applications_ini['Application']['Aspect'] = '1'
        elif ratio >= 4 / 3:
            applications_ini['Application']['Aspect'] = '0'
        else:
            applications_ini['Application']['Aspect'] = '2'
    elif surface_shape == 'circle':
        applications_ini['Application']['Aspect'] = '2'  # circle or square
        applications_ini['ProjOffset']['shape'] = 'True'
    else:  # floor
        applications_ini['Application']['Aspect'] = '0'  # doesn't really matter
    applications_ini['ProjOffset']['Aspect'] = applications_ini['Application']['Aspect']
    applications_ini['Application']['RoundMask'] = applications_ini['ProjOffset']['shape']


    applications_ini['ProjOffset']['xOffset'] = '%f' % (xOffset / calibrator.screen_width - .5)
    applications_ini['ProjOffset']['yOffset'] = '%f' % (yOffset / calibrator.screen_height - .5)
    applications_ini['Application']['xOffset'] = '%f' % (xOffset / calibrator.screen_width - .5)
    applications_ini['Application']['yOffset'] = '%f' % (yOffset / calibrator.screen_height - .5)

    with open(applications_ini_path, 'w') as configfile:
        applications_ini.write(configfile, space_around_delimiters=False)


def export_to_eyeclick_ini(calibrator, xOffset, yOffset):
    eyeclick_ini = get_ini_as_list(eyeclick_ini_path)
    eyeclick_ini['ProjOffset']['xOffset'] = '%f' % (xOffset / calibrator.screen_width - .5)
    eyeclick_ini['ProjOffset']['yOffset'] = '%f' % (yOffset / calibrator.screen_height - .5)
    with open(eyeclick_ini_path, 'w') as configfile:
        eyeclick_ini.write(configfile, space_around_delimiters=False)


def export_to_eyestep_ini(calibrator):
    eyestep_ini = get_ini_as_list(eyestep_ini_path)
    top_right, top_left, bottom_right, bottom_left = get_4_corners_of_screen_in_depthmap_upright(calibrator)
    camera_img_width = 320
    camera_img_height = 240
    eyestep_ini['Zone1']['zoneVertex1'] = "%f,%f" % (
        bottom_left[0, 0, 0] / camera_img_width, bottom_left[0, 0, 1] / camera_img_height)
    eyestep_ini['Zone1']['zoneVertex2'] = "%f,%f" % (
        top_left[0, 0, 0] / camera_img_width, top_left[0, 0, 1] / camera_img_height)
    eyestep_ini['Zone1']['zoneVertex3'] = "%f,%f" % (
        top_right[0, 0, 0] / camera_img_width, top_right[0, 0, 1] / camera_img_height)
    eyestep_ini['Zone1']['zoneVertex4'] = "%f,%f" % (
        bottom_right[0, 0, 0] / camera_img_width, bottom_right[0, 0, 1] / camera_img_height)
    if calibrator.projection_flip_type == 'ud' or calibrator.projection_flip_type == 'lr_ud':
        eyestep_ini['Camera1']['flipud'] = 'True'
    else:
        eyestep_ini['Camera1']['flipud'] = 'False'
    if calibrator.projection_flip_type == 'lr' or calibrator.projection_flip_type == 'lr_ud':
        eyestep_ini['Camera1']['fliplr'] = 'True'
    else:
        eyestep_ini['Camera1']['fliplr'] = 'False'
    with open(eyestep_ini_path, 'w') as configfile:
        eyestep_ini.write(configfile, space_around_delimiters=False)


def get_ini_as_list(ini_path):
    ini = configparser.ConfigParser()
    ini.optionxform = str
    if os.path.isfile(applications_ini_path):
        ini.read(ini_path)
    return ini


def get_4_corners_of_screen_in_depthmap(calibrator):
    bottom_left, bottom_right, top_left, top_right = calibrator.get_display_corners_on_cam()
    return top_right, top_left, bottom_right, bottom_left


def get_4_corners_of_screen_in_depthmap_upright(calibrator):
    corners = np.squeeze(np.concatenate(get_4_corners_of_screen_in_depthmap(calibrator), axis=0), axis=1)[:,:2]
    if calibrator.projection_flip_type == 'ud' or calibrator.projection_flip_type == 'lr_ud':
        corners[:,1] = 240-corners[:,1]
    if calibrator.projection_flip_type == 'lr' or calibrator.projection_flip_type == 'lr_ud':
        corners[:, 0] = 320 - corners[:, 0]
    return corners[0, :].reshape(1, 1, -1), corners[1, :].reshape(1, 1, -1), \
           corners[2, :].reshape(1, 1, -1), corners[3, :].reshape(1, 1, -1)


