import base64
import glob
import os

from pyeyeengine.calibration.auto_calibrator import GRID_IMAGES_SAVE_PATH, CalibrationPatternCollector, AutoCalibrator


def _calibrate(calibrator):
    calibrator.calibrate(recollect_imgs=False, mode="floor")

def _write_image_pairs(image_pairs):
    if not os.path.exists(GRID_IMAGES_SAVE_PATH):
        os.mkdir(GRID_IMAGES_SAVE_PATH)
    _clear_image_grid_dir()
    for i, image_pair in enumerate(image_pairs):
        _write_image_pair(i, image_pair)

def _get_calibration_images_grid(request):
    if not os.path.exists(GRID_IMAGES_SAVE_PATH):
        os.mkdir(GRID_IMAGES_SAVE_PATH)
    _clear_image_grid_dir()
    if "chessboard_grid_gaps" in request["params"].keys():
        CalibrationPatternCollector(request["params"]["resolution"],
                    chessboard_grid_gaps=request["params"]["chessboard_grid_gaps"]).save_chessboard_patters_pngs()
    else:
        CalibrationPatternCollector(request["params"]["resolution"]).save_chessboard_patters_pngs()
    return glob.glob(GRID_IMAGES_SAVE_PATH + "displayed*.png")

def _clear_image_grid_dir():
    [os.remove(os.path.join(GRID_IMAGES_SAVE_PATH, file)) for file in os.listdir(GRID_IMAGES_SAVE_PATH)]


def _write_image_pair( image_index, image_pair):
    _write_image("%s_%d.png" % ("displayed", image_index), base64.b64decode(image_pair["displayed"]))
    _write_image("%s_%d.png" % ("viewed", image_index), base64.b64decode(image_pair["viewed"]))


def _write_image(file_name, image_bytes):
    with open(os.path.join(GRID_IMAGES_SAVE_PATH, file_name), "wb") as f:
        f.write(image_bytes)

