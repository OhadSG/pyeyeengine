import os
import cv2
import time
from datetime import datetime
import subprocess as sp
from pyeyeengine.utilities import global_params as Globals
from pyeyeengine.camera_utils.frame_manager import FrameManager
from pyeyeengine.utilities.logging import Log

# FUNCTIONS

FUNCTION_FOCUS = "FOCUS"

# CONSTANTS

PROJECTOR_DRIVER_PATH = "/etc/epscom-cmd"
BLUR_MIN_THRESHOLD = 320.0
AUTO_FOCUS_LOCKS = 10
MAX_AUTO_FOCUS_TIME = 40.0 # seconds
FLOW = "auto_focus"

# IMPLEMENTATION

def __variance_of_laplacian(image):
    try:
        return cv2.Laplacian(image, cv2.CV_64F).var()
    except Exception as e:
        Log.e("Error trying to get Laplacian value of image", extra_details={"error": "{}".format(e)}, flow=FLOW)
        return None

def supports_function(function):
    p = sp.Popen([PROJECTOR_DRIVER_PATH, function + "?"], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    output, catch = p.communicate(b"input data that is passed to subprocess' stdin")

    if "FOCUS" in output.decode("utf-8"):
        return True
    else:
        return False

def send_function(function, param=None):
    if param is not None:
        p = sp.Popen([PROJECTOR_DRIVER_PATH, function, param], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    else:
        p = sp.Popen([PROJECTOR_DRIVER_PATH, function], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)

    output, catch = p.communicate(b"input data that is passed to subprocess' stdin")

    if not "ERR" in output.decode("utf-8"):
        return True
    else:
        return False

def reset_focus():
    assert supports_function("FOCUS")
    send_function("FOCUS MIN")

def change_focus(mode=Globals.Commons.decrease, amount=0):
    assert supports_function("FOCUS")

    for i in range(amount):
        if mode is Globals.Commons.decrease:
            send_function("FOCUS DEC")
        elif mode is Globals.Commons.increase:
            send_function("FOCUS INC")
        else:
            pass
        time.sleep(0.05)

def __get_current_laplacian_score():
    current_frame = FrameManager.getInstance().rgb_stream.get_frame()
    margin = 250
    current_frame = current_frame[margin:-margin, margin:-margin]
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    return __variance_of_laplacian(gray)

def __get_average_laplacian(samples = 5):
    current_frame = FrameManager.getInstance().get_rgb_frame()
    margin = 250
    current_frame = current_frame[margin:-margin, margin:-margin]
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    scores = []

    for i in range(samples):
        scores.append(__variance_of_laplacian(gray))

    return (sum(scores) / len(scores))

def __perform_auto_focus():
    reset_focus()
    time.sleep(2)
    FrameManager.getInstance().set_rgb_resolution(Globals.RGB_HIGH_QUALITY)
    auto_focus_start_time = datetime.now()
    trend = "up"
    focus_samples = []
    did_find_peak = False
    down_counter = 0
    down_max = 5
    steps = 1

    while not did_find_peak:
        current_time = datetime.now()
        delta_time = (current_time - auto_focus_start_time).total_seconds()

        if delta_time > MAX_AUTO_FOCUS_TIME:
            return False, "Exceeded time limit"

        current_score = __get_average_laplacian(5)

        Log.d("Current score", extra_details={"current": current_score})

        if len(focus_samples) < 2:
            Log.d("Not enough samples, increasing focus", flow=FLOW)
            change_focus(Globals.Commons.increase, steps)
            focus_samples.append(current_score)
        elif current_score > focus_samples[len(focus_samples) - 1]:
            Log.d("Current score is better, increasing focus", extra_details={"current": current_score, "previous": focus_samples[len(focus_samples) - 1]}, flow=FLOW)
            trend = "up"
            down_counter = 0
            change_focus(Globals.Commons.increase, steps)
            focus_samples.append(current_score)
        else:
            Log.d("Current score is worse, still increasing focus", extra_details={"current": current_score, "previous": focus_samples[len(focus_samples) - 1]}, flow=FLOW)
            trend = "down"
            down_counter += 1
            change_focus(Globals.Commons.increase, steps)
            focus_samples.append(current_score)

        Log.d("Current focus trend", extra_details={"trend": trend})

        if down_counter is down_max:
            change_focus(Globals.Commons.decrease, down_max * steps)
            did_find_peak = True
            return True, __get_average_laplacian()

    return False, "Unknown"

def auto_focus():
    Log.i("Auto-focusing Projection", flow=FLOW)

    try:
        did_succeed, message = __perform_auto_focus()
    except Exception as e:
        Log.e("Auto-focus exception", extra_details={"exception": "{}".format(e)}, flow=FLOW)
        return False

    if did_succeed:
        Log.i("Auto-focus Done", extra_details={"focus_quality": "{:.2f}".format(message)}, flow=FLOW)
    else:
        Log.e("Auto-focus failed", extra_details={"error": message}, flow=FLOW)

    return did_succeed

# def legacy():
#     previous_largest = 0
#     largest_value = 0
#     previous_value = 0
#     locks = 0
#     steps = 2
#
#     auto_focus_start_time = datetime.now()
#
#     reset_focus()
#     current_focus_val = 0
#     time.sleep(2)
#
#     FrameManager.getInstance().set_rgb_resolution(Globals.RGB_HIGH_QUALITY)
#
#     while locks < AUTO_FOCUS_LOCKS:
#         blur_intensity = __get_current_laplacian_score()
#
#         if blur_intensity is None:
#             Log.d("Got a bad result", extra_details={"blur_intensity": blur_intensity}, flow=FLOW)
#             return False
#
#         previous_largest = largest_value
#
#         if (blur_intensity - largest_value) > 1:
#             Log.d("Got a better Lap. result",
#                   extra_details={"new_result": blur_intensity, "previous_largest": largest_value}, flow=FLOW)
#             largest_value = blur_intensity
#             locks = 0
#
#         # Log.i("Int: {} PrevVal: {} LargestVal: {} PrevLargest: {} Locks: {}".format(blur_intensity, previous_value, largest_value, previous_largest, locks), flow=FLOW)
#
#         current_time = datetime.now()
#         delta_time = (current_time - auto_focus_start_time).total_seconds()
#
#         if locks == 0 and delta_time > MAX_AUTO_FOCUS_TIME:
#             Log.e("Auto-focus exceeded time limit, aborting", flow=FLOW)
#             locks = AUTO_FOCUS_LOCKS
#             return False
#
#         Log.d("Current Largest: {} Current Value: {}".format(largest_value, blur_intensity))
#
#         if (largest_value - blur_intensity) > 1 and largest_value > BLUR_MIN_THRESHOLD:
#             Log.d("Found largest value, locking", extra_details={"largest_value": largest_value}, flow=FLOW)
#             change_focus(Globals.Commons.decrease, steps * 2)
#             current_focus_val -= steps * 2
#             locks = AUTO_FOCUS_LOCKS
#         elif (previous_value - blur_intensity) > 1:  # if previous lap. score was higher than current (by at least 1)
#             Log.d("Worse result, decreasing...",
#                   extra_details={"lap_result": blur_intensity, "previous_result": previous_value}, flow=FLOW)
#             change_focus(Globals.Commons.decrease, steps)
#             current_focus_val -= steps
#         else:  # current lap. score is higher than previous, keep raising focus
#             Log.d("Adjusting focus", extra_details={"lap_result": blur_intensity}, flow=FLOW)
#             change_focus(Globals.Commons.increase, steps)
#             current_focus_val += steps
#
#         previous_value = blur_intensity
#
#         if previous_largest == largest_value:
#             locks += 1
#
#         if locks is AUTO_FOCUS_LOCKS and largest_value < BLUR_MIN_THRESHOLD:
#             locks = 0
#
#     Log.i("Auto-focus Done", extra_details={"focus_quality": "{:.2f}".format(__get_current_laplacian_score())},
#           flow=FLOW)
#
#     return True