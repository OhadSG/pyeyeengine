import os
import ctypes
import platform
from primesense import openni2
from primesense import _openni2 as c_api
from primesense._openni2 import OniVersion
from pyeyeengine.utilities.init_openni import init_openni

if __name__ == '__main__':
    init_openni()
    device = openni2.Device.open_any()
    hardware_version = device.get_int_property(c_api.ONI_DEVICE_PROPERTY_HARDWARE_VERSION)
    firmware_version = device.get_property(c_api.ONI_DEVICE_PROPERTY_FIRMWARE_VERSION,
                                                (ctypes.c_char * 100))
    driver_version = device.get_property(c_api.ONI_DEVICE_PROPERTY_DRIVER_VERSION, OniVersion)
    serial_number = device.get_property(c_api.ONI_DEVICE_PROPERTY_SERIAL_NUMBER,
                                             (ctypes.c_char * 100))

    print({"hardware_version": hardware_version,
            "firmware_version": firmware_version.value.decode("utf-8"),
            "driver_version": "{}.{}.{}.{}".format(driver_version.major, driver_version.minor,
                                                   driver_version.maintenance, driver_version.build),
            "serial_number": serial_number.value.decode("utf-8")})