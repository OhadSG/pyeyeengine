import time
import sys
import numpy as np

import os
import cv2
from tkinter import messagebox
import subprocess

from pyeyeengine.server.screen_setter import WinScreenSetter

'''
instructions:
1) open pycharm with admin rights.
2) switch python interpeter to pyeyeengine locally on pc. 
3) try to run and debug - may have to comment out __init__.py
4) go to terminal in pycharm : pyinstaller pyeyeengine/auto_calibrate.py
5) add needed parts of pyeyeengine - for example openni folder - in the same folder hierarchy as expected. 
'''

# from eye_engine.eye_engine import EyeEngine
from pyeyeengine.calibration.auto_calibrator import AutoCalibrator
from pyeyeengine.calibration.export_calibration import export_data_for_games_and_cpp_engine
from pyeyeengine.calibration.find_max_playing_mask import find_max_playing_mask
from pyeyeengine.camera_utils.frame_manager import FrameManager

stand_clear_img = cv2.putText(np.ones((240, 320, 3), 'uint8') * 255, "Please stand away ",
                              (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
stand_clear_img = cv2.putText(stand_clear_img, "from the white",
                              (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
stand_clear_img = cv2.putText(stand_clear_img, "projection",
                              (20, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))

stand_clear_img = cv2.putText(stand_clear_img, "Please stand away ",
                              (20, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
stand_clear_img = cv2.putText(stand_clear_img, "from the white",
                              (20, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
stand_clear_img = cv2.putText(stand_clear_img, "projection",
                              (20, 220), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

cv2.namedWindow("stand_clear", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("stand_clear", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("stand_clear", stand_clear_img)
cv2.waitKey(1000)
os.system("Taskkill /IM eyeEngine3d.exe /F")
time.sleep(1)
cv2.destroyAllWindows()

calibrator = AutoCalibrator(FrameManager.getInstance())
calibrator.calibrate(recollect_imgs=True, screen_setter=WinScreenSetter(screen_height=calibrator.screen_height,
                                                                        screen_width=calibrator.screen_width))

rbg_low_res = cv2.imwrite("./rbg_low_res.png", calibrator._camera.get_rgb(res_xy=(320, 240)))
rgb_medium_res = cv2.imwrite("./rbg_low_res.png", calibrator._camera.get_rgb(res_xy=(640, 480)))
rgb_high_res = cv2.imwrite("./rbg_low_res.png", calibrator._camera.get_rgb(res_xy=(1280, 960)))
if calibrator.calibrate_success:
    playing_screen_mask = find_max_playing_mask(calibrator, is_display=False)
    export_data_for_games_and_cpp_engine(calibrator)
    # eye_engine = EyeEngine()
    # while True:
    #     eye_engine.display_contours_on_rbg([], [], eye_engine._frame_grabber.get_rgb(), calibrator)
    # messagebox.showinfo("result", "Calibrated successfully. Please check that games work properly to make sure.")
    print("Calibrated successfully. Please check that games work properly to make sure.")
    # cv2.waitKey(0)
    calibrator._camera._camera.stop()
    subprocess.Popen(["C:\Windows\System32\EyeclickShell.exe"])
    time.sleep(10)
    sys.exit(0)
else:
    messagebox.showinfo("result", "Calibration failed. If this is your first try, please try again.")
    sys.exist(1)
