import os
import subprocess
import time
import traceback

import numpy as np
import usb.core


# https://files.support.epson.com/pdf/pl600p/pl600pcm.pdf


class ProjectorController_pyusb:
    def __init__(self) -> None:
        super().__init__()
        self.EPSON_DEVICE_VENDOR = 0x04b8
        self.EPSON_DEVICE_PRODUCT = 0x0514
        self._device = usb.core.find(idVendor=self.EPSON_DEVICE_VENDOR, idProduct=self.EPSON_DEVICE_PRODUCT)

    def turn_on(self):
        self._send_message_to_projector("PWR ON")

    def turn_off(self):
        self._send_message_to_projector("PWR OFF")

    def change_vertical_keystone(self, degree):
        deg_val = np.minimum(degree / 30 * 123 + 123, 255) if degree > 0 else np.maximum(0, 123 + degree / 20 * 123)
        self._send_message_to_projector("VKEYSTONE %d" % deg_val)

    def _send_message_to_projector(self, message):
        self._device.reset()
        time.sleep(1)
        self._device.write(self._device[0][(2, 0)][0], ("%s\r" % message).encode())
        time.sleep(1)


class ProjectorController:
    def __init__(self):
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.EPSCOM_CMD = file_path + '/epscom-driver/epscom-cmd'

    def full_command(self, cmd):
        return "%s \"%s\"" % (self.EPSCOM_CMD, cmd)

    def epscom_set(self, cmd):
        subprocess.call(self.full_command(cmd), shell=True)
        time.sleep(2)

    def epscom_get(self, cmd):
        time.sleep(2)
        try:
            try:
                answer = subprocess.check_output(self.full_command(cmd), shell=True)
                return self.clean_returned_byte_string(answer)
            except subprocess.CalledProcessError as e:
                return self.clean_returned_byte_string(e.output)
        except subprocess.CalledProcessError as e:
            return str("projector cmd failed: %s" % (cmd))

    def clean_returned_byte_string(self, answer):
        return answer.decode("utf-8").replace("\n", "").split("=")[-1]

    def is_turned_on(self):
        return self.epscom_get("PWR?")

    def turn_on(self):
        self.epscom_set("PWR ON")

    def turn_off(self):
        self.epscom_set("PWR OFF")

    def set_view_ceiling(self, view="ON"):
        self.epscom_set("VREVERSE %s" % view)

    def set_view_rear(self, view="ON"):
        self.epscom_set("HREVERSE %s" % view)

    def get_view_ceiling(self):
        return self.epscom_get("VREVERSE?")

    def get_view_rear(self):
        return self.epscom_get("HREVERSE?")

    def set_auto_keystones(self, mode="ON"):
        assert mode == "ON" or mode == "OFF"
        return self.epscom_set("AUTOKEYSTONE %s" % mode)

    def check_is_auto_keystones_on(self):
        return self.epscom_get("AUTOKEYSTONE?")

    def get_vertical_keystone(self):
        vertical_keystone = self.epscom_get("VKEYSTONE?")
        return ((np.float32(vertical_keystone) - 128) / 128) * 30

    def get_horizontal_keystone(self):
        horizontal_keystones = self.epscom_get("HKEYSTONE?")
        return ((np.float32(horizontal_keystones) - 128) / 128) * 20

    def change_vertical_keystone(self, degree):
        max_angle = 37.736
        min_angle = -13.134
        degree = np.minimum(np.maximum(degree, min_angle), max_angle)
        deg_val = np.minimum(degree / max_angle * 127 + 128, 255) \
            if degree > 0 else np.maximum(0, 128 + (degree / np.abs(min_angle)) * 127)
        self.epscom_set("VKEYSTONE %d" % deg_val)

    def change_horizontal_keystone(self, degree):
        max_angle = 15
        degree = np.minimum(np.maximum(degree, -max_angle), max_angle)
        deg_val = np.minimum(degree / max_angle * 127 + 128, 255) \
            if degree > 0 else np.maximum(0, 128 + degree / max_angle * 127)
        self.epscom_set("HKEYSTONE %d" % deg_val)

    def get_lamp_hours(self):
        return self.epscom_get("LAMP?")

    def get_serial_number(self):
        return self.epscom_get("SNO?")

    def get_temp(self):
        return self.epscom_get("TEMP?")

    def set_aspect_to_full(self):
        self.epscom_set("ASPECT %d" % 40)

    def is_aspect_full(self):
        return self.epscom_get("ASPECT?") == '40'

    def set_dynamic_color_mode(self):
        return self.epscom_set("CMODE %d" %6)

    def is_dynamic_color_mode(self):
        return self.epscom_get("CMODE?") == "06"


if __name__ == '__main__':
    # hrs = int(ProjectorController().get_lamp_hours())
    # assert (hrs > 0)

    out = ProjectorController().get_temp()
    hi = 5
