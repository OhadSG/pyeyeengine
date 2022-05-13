import cv2
import time


# calibrate
from pyeyeengine.calibration.auto_calibrator import AutoCalibrator
from pyeyeengine.calibration.find_max_playing_mask import find_max_playing_mask
from pyeyeengine.camera_utils.camera_reader import CameraReader
from pyeyeengine.eye_engine.eye_engine import EyeEngine

calibrator = AutoCalibrator()
calibrator.calibrate()
if calibrator.calibrate_success:
    playing_screen_mask = find_max_playing_mask(calibrator, display=True)
    # export_data_for_games_and_cpp_engine(calibrator)

# os.system("start C:\\Users\\EyePlay\\Documents\\EyeclickShell.exe.lnk")
# quit()
# run engine
cam = CameraReader(display=False, register=True)
engine = EyeEngine()

is_calibrated = True
keep_running = True

while keep_running:
    key = cv2.waitKey(1)
    if key == 27:
        keep_running = False

    start_time = time.time() * 1000
    dmap = cam.get_depth()
    rgb = cam.get_rgb()
    end_time = time.time() * 1000
    print("camera_reading_time: %f" %(end_time-start_time))

    if is_calibrated:
        engine.process_frame(dmap, rgb, calibrator, display=True)
        calibrator.display_hands_with_homography(engine.hands, playing_screen_mask)
## Release resources
cv2.destroyAllWindows()
cam.stop()
