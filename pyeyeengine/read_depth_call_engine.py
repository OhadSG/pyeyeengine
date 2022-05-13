import cv2

# from cnn_finger_localization.eye_engine import EyeEngine
import numpy as np
from pyeyeengine.calibration.export_calibration import export_data_for_android

from pyeyeengine.eye_engine.light_touch_engine import LightEyeEngine
from pyeyeengine.object_detection.key_points_extractor import PointingFingerExtractor
from pyeyeengine.object_detection.key_points_extractor import FootEdgeExtractor
from pyeyeengine.calibration.auto_calibrator import AutoCalibrator
from pyeyeengine.eye_engine.eye_engine import EyeEngine
# from pyeyeengine.object_detection.ML_key_points_extractor import PointingFingerMLExtractor

# from cnn_finger_localization.eye_engine import EyeEngine


# engine = LightEyeEngine()
engine = EyeEngine(key_pts_extractor=PointingFingerExtractor())
# engine._calibrator.set_warp_cam_to_display(np.array([-8.073620863224027, -0.39849598348404097, 2218.927804428893,
#                                                 -0.06909563209881509, 7.152040784650158, -461.1141748103403,
#                                                 -8.860870409713614E-6, -3.6763877623197935E-4, 1.0]).reshape(3,3))#np.eye(3)
# engine._calibrator.get_table_mask()
engine._calibrator.load_table_data()
mask_for_display, rect_display = export_data_for_android(engine._calibrator)

keep_running = True
is_calibrated = False
frame_time = 30
while keep_running:
    key = cv2.waitKey(1)
    if key == 27:
        keep_running = False
    # start_time = time.clock()
    # print(time.clock()-start_time)

    engine.process_frame(show_time=True, display=False)


    # print(engine.tracker.get_key_points())
    # print(calibrator.transfrom_points_cam_to_display(engine.tracker.get_key_points()))

    # cv2.imshow('depth', np.uint8(depth_map*.05))
    # cv2.imshow('rbg', rgb)

## Release resources
cv2.destroyAllWindows()


