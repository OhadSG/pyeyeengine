import numpy as np
import cv2
import time
import os
from pyeyeengine.camera_utils.camera_manager import CameraManager
from pyeyeengine.utilities.astra_orbbec_conf import AstraOrbbec

camera_conf = AstraOrbbec(scale_factor = 1) #scale factor to maximum resolution of 480 x 640.



frame_grabber = CameraManager()
rgb_enlarged = frame_grabber.get_rgb(res_xy=(1280, 960))

######## Display depth_map, background and diff_map


while True:
    #get depth map

    depth_map = frame_grabber.get_depth(res_xy=(camera_conf.y_res, camera_conf.x_res))
    rgb_image = frame_grabber.get_rgb(res_xy=(camera_conf.y_res, camera_conf.x_res))

    #depth_map_adj = np.uint16(((depth_map - min) / (max- min)) * 65535)
    #scale pixel values for depth map
    max_depth_map = np.max(depth_map)
    min_depth_map = np.min(depth_map)
    depth_map_adj = np.uint8(((depth_map - min_depth_map) / (np.max((max_depth_map - min_depth_map, 1)))) * 255)

    #depth_map_adj = cv2.Sobel(depth_map_adj, cv2.CV_8UC1, 1,0,ksize=5)
    #depth_map_adj = cv2.Sobel(depth_map_adj, cv2.CV_8UC1, 1, 0, ksize=5)

    depth_map_adj = cv2.applyColorMap(depth_map_adj, cv2.COLORMAP_JET)
    #rgb_image = cv2.applyColorMap(rgb_image, cv2.COLORMAP_JET)

    #rgb_image = np.uint8((rgb_image + depth_map_adj) / 2)

    cv2.imshow('depth_map', depth_map_adj)
    cv2.imshow('rgb_image', rgb_image)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.ioff()
plt.show()
