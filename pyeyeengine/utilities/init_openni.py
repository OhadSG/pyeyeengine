from primesense import openni2
import os
import logging

logger = logging.getLogger(__name__)

def init_openni():
    logger.info("Initializing OpenNI driver")

    openni2.initialize(__get_driver_path())


def __get_driver_path():
    path = os.getenv('OPENNI2_REDIST')
    if path is not None:
        return path

    distribution = "/usr/local/lib/python3.5/dist-packages/pyeyeengine/OpenNI-Linux-Arm-2.3/Redist"
    return distribution

    # if "arm" in platform.machine():
    #     distribution += "_ARM"
    # elif "Linux" in platform.system():
    #     distribution += "_Linux_64"
    # elif "Darwin" in platform.system():
    #     distribution += "_Mac_64"
    # else:
    #     if platform.architecture()[0] == "32bit":
    #         distribution += "_WIN_32"
    #     else:
    #         distribution += "_WIN_64"
    #
    # return distribution