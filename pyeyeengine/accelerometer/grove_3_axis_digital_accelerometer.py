import smbus
import time
import numpy as np
from matplotlib import pyplot as plt

bus = smbus.SMBus(4)

# ADXL345 device address
ADXL345_DEVICE = 0x53

# ADXL345 constants
EARTH_GRAVITY_MS2   = 9.80665
SCALE_MULTIPLIER    = 0.004

DATA_FORMAT         = 0x31
BW_RATE             = 0x2C
POWER_CTL           = 0x2D

BW_RATE_1600HZ      = 0x0F
BW_RATE_800HZ       = 0x0E
BW_RATE_400HZ       = 0x0D
BW_RATE_200HZ       = 0x0C
BW_RATE_100HZ       = 0x0B
BW_RATE_50HZ        = 0x0A
BW_RATE_25HZ        = 0x09

RANGE_2G            = 0x00
RANGE_4G            = 0x01
RANGE_8G            = 0x02
RANGE_16G           = 0x03

MEASURE             = 0x08
AXES_DATA           = 0x32

class ADXL345:

    address = None

    def __init__(self, address = ADXL345_DEVICE):
        self.address = address
        self.setBandwidthRate(BW_RATE_100HZ)
        self.setRange(RANGE_2G)
        self.enableMeasurement()

    def enableMeasurement(self):
        bus.write_byte_data(self.address, POWER_CTL, MEASURE)

    def setBandwidthRate(self, rate_flag):
        bus.write_byte_data(self.address, BW_RATE, rate_flag)

    # set the measurement range for 10-bit readings
    def setRange(self, range_flag):
        value = bus.read_byte_data(self.address, DATA_FORMAT)

        value &= ~0x0F;
        value |= range_flag;
        value |= 0x08;

        bus.write_byte_data(self.address, DATA_FORMAT, value)

    # returns the current reading from the sensor for each axis
    #
    # parameter gforce:
    #    False (default): result is returned in m/s^2
    #    True           : result is returned in gs
    def getAxes(self, gforce = False):
        bytes = bus.read_i2c_block_data(self.address, AXES_DATA, 6)

        x = bytes[0] | (bytes[1] << 8)
        if(x & (1 << 16 - 1)):
            x = x - (1<<16)

        y = bytes[2] | (bytes[3] << 8)
        if(y & (1 << 16 - 1)):
            y = y - (1<<16)

        z = bytes[4] | (bytes[5] << 8)
        if(z & (1 << 16 - 1)):
            z = z - (1<<16)

        x = x * SCALE_MULTIPLIER
        y = y * SCALE_MULTIPLIER
        z = z * SCALE_MULTIPLIER

        if gforce == False:
            x = x * EARTH_GRAVITY_MS2
            y = y * EARTH_GRAVITY_MS2
            z = z * EARTH_GRAVITY_MS2

        x = round(x, 4)
        y = round(y, 4)
        z = round(z, 4)

        return {"x": x, "y": y, "z": z}

    def getAngles(self):
        axes = self.getAxes(True)
        x = axes['x']
        y = axes['y']
        z = axes['z']

        theta = np.degrees(np.arctan(x / np.sqrt(y**2 + z**2)))
        psi = np.degrees(np.arctan(y / np.sqrt(x ** 2 + z ** 2)))
        fi = np.degrees(np.arctan(z / np.sqrt(x ** 2 + y ** 2)))

        return theta, psi, fi




if __name__ == "__main__":
    # if run directly we'll just create an instance of the class and output
    # the current readings
    adxl345 = ADXL345()

    psi_list = []
    while True:
        axes = adxl345.getAxes(True)
        x = axes['x']
        y = axes['y']
        z = axes['z']

        theta = np.degrees(np.arctan(x / np.sqrt(y**2 + z**2)))
        psi = np.degrees(np.arctan(y / np.sqrt(x ** 2 + z ** 2)))
        fi = np.degrees(np.arctan(z / np.sqrt(x ** 2 + y ** 2)))

        print("ADXL345 on address 0x%x:" % (adxl345.address))
        print("   x = %.3fG" % ( axes['x'] ))
        print("   y = %.3fG" % ( axes['y'] ))
        print("   z = %.3fG" % ( axes['z'] ))

        print("   theta = %.3f degrees" % ( theta ))
        print("   psi = %.3f degrees" % ( psi ))
        print("   fi = %.3f degrees" % ( fi ))


        psi_list.append(psi)

        print(len(psi_list))

        if len(psi_list) > 50:
            print("Angle: ",  sum(psi_list)/len(psi_list))
            plt.plot(psi_list)
            plt.show()
            break


        time.sleep(2)

