from pyeyeengine.utilities import logging
from pyeyeengine.accelerometer.grove_3_axis_digital_accelerometer import ADXL345
import time

adxl345 = ADXL345()

while True:

    winkel_list = []
    for i in range(100):
        winkel1 , winkel2, _ = adxl345.getAngles()
        winkel_list.append(winkel2)

    obie_angle = sum(winkel_list)/len(winkel_list)
    print(obie_angle)

