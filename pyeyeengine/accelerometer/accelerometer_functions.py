from pyeyeengine.accelerometer.grove_3_axis_digital_accelerometer import ADXL345
from pyeyeengine.utilities import global_params

def get_total_angle_obie(iterations = 50):
    adxl345 = ADXL345() #accelerometer class
    angle_list = []
    for i in range(iterations): #average over #number iterations of angles.
        angle, _, _ = adxl345.getAngles()
        angle_list.append(angle)

    obie_angle = sum(angle_list) / len(angle_list)
    camera_angle = global_params.CAMERA_ANGLE_OBIE
    return camera_angle - obie_angle #return the total angle