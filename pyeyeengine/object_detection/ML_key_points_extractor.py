import cv2

import numpy as np

# from keras.models import load_model

from pyeyeengine.object_detection.key_points_extractor import get_centroids, find_pointing_finger

# finger_detector_convnet = load_model("./object_detection/finger_detctor_light.h5")
finger_detector_convnet = cv2.dnn.readNetFromTensorflow('./object_detection/finger_detctor_light.pb')

def round_int(num):
    return np.int32(np.round(num))


def crop_and_pad(img, bbox):
    x1, y1, x2, y2 = [round_int(pt) for pt in bbox]
    img_croped = img[np.maximum(0, y1): np.minimum(img.shape[0], y2),
                 np.maximum(0, x1): np.minimum(img.shape[1], x2), :]
    return np.pad(img_croped, ((np.maximum(0, -y1), np.maximum(0, -(img.shape[0] - y2))),
                               (np.maximum(0, -x1), np.maximum(0, -(img.shape[1] - x2))), (0, 0)), mode="constant")


def ml_correct_finger_loc(finger_loc_xy, rgb, radius=15):
    bbox = np.concatenate([finger_loc_xy.reshape(-1) - radius, finger_loc_xy.reshape(-1) + radius + 1], axis=0)
    window = crop_and_pad(rgb, bbox)
    window_blob = cv2.dnn.blobFromImage(window)
    finger_detector_convnet.setInput(window_blob)
    return finger_detector_convnet.forward().reshape(1, -1) + np.array([bbox[:2]])
    # return finger_detector_convnet.predict(np.expand_dims(window, axis=0)).reshape(1, -1) + \
    #        np.array([bbox[:2]])

class PointingFingerMLExtractor:
    def __init__(self):
        self.max_height_above_playing_surface = 50

    def extract(self, voxels_list,  frame_grabber=None):
        rgb = frame_grabber.get_rgb()
        return get_centroids(voxels_list), \
               [ml_correct_finger_loc(find_pointing_finger(voxels), rgb) for voxels in voxels_list]
