import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os
from pyeyeengine.eye_engine.eye_engine import EyeEngine


engine = EyeEngine()


######## Display depth_map, background and diff_map


while True:
    #get depth map
    start_time = time.time()
    depth_map = cv2.resize(engine._frame_grabber.get_depth(res_xy=(320, 240)), (320, 240))
    end_time = time.time()
    print('time for calculation matrixwise: ', end_time - start_time)

    #update background model and get it's value
    engine._background_model.update_background_model(depth_map, engine._object_detector.get_binary_objects(), engine._calibrator.table_mask)
    bkgr = np.round(engine._background_model.background_model_adjusted)

    #gett diff_map
    diff_map = engine._background_model.detect_change(depth_map)


    #depth_map_adj = np.uint16(((depth_map - min) / (max- min)) * 65535)
    #scale pixel values for depth map
    max_depth_map = np.max(depth_map)
    min_depth_map = np.min(depth_map)
    depth_map_adj = np.uint8(((depth_map - min_depth_map) / (np.max((max_depth_map - min_depth_map, 1)))) * 255)

    depth_map_adj = cv2.applyColorMap(depth_map_adj, cv2.COLORMAP_JET)

    #scale pixel values for bkgr
    max_bkgr = np.max(bkgr)
    min_bkgr = np.min(bkgr)
    bkgr_adj = np.uint8(((bkgr - min_bkgr) / (np.max((max_bkgr - min_bkgr, 1)))) * 255)

    bkgr_adj = cv2.applyColorMap(bkgr_adj, cv2.COLORMAP_JET)

    #scale pixel values for diff_map
    max_diff_map = np.max(diff_map)
    min_diff_map = np.min(diff_map)
    diff_map_adj = np.uint8(((diff_map - min_diff_map) / (np.max((max_diff_map - min_diff_map, 1)))) * 255)

    diff_map_adj = cv2.applyColorMap(diff_map_adj, cv2.COLORMAP_JET)


    cv2.imshow('depth_map', depth_map_adj)
    cv2.imshow('background', bkgr_adj)
    cv2.imshow('diff_map', diff_map_adj)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.ioff()
plt.show()
