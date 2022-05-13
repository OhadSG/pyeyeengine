import time

import cv2
import numpy as np


class ScreenSetter(object):
    def __init__(self, socketio, screen_width=1280, screen_height=800):
        self._image_to_show = None
        self.set_screen_res(screen_width=screen_width, screen_height=screen_height)
        self.camera = None
        self.top = 0
        self.left = 0
        self.has_image_been_shown = False
        self.socketio = socketio

    def get_frame(self):
        if self.camera is None:
            ret, png = cv2.imencode('.png', self._image_to_show)
        else:
            ret, png = cv2.imencode('.png', self.camera.get_rgb())
        self.has_image_been_shown = True
        return png.tobytes()

    def get_image_to_show(self):
        self.has_image_been_shown = True
        return self._image_to_show

    def set_image(self, img, top=0, left=0):
        self._image_to_show = img.copy()
        self.set_image_top_left(top, left)
        self.socketio.emit('replace_image_data', {'src': "http://127.0.0.1:2222/get_image"})
        self.has_image_been_shown = False

    def set_image_top_left(self, top, left):
        if self.top != top or self.left != left:
            self.top, self.left = top, left
            self.socketio.emit('replace_image_data', {'top': top, 'left': left})

    def set_screen_res(self, screen_width, screen_height):
        self._image_to_show = np.uint8(np.ones((screen_width, screen_height, 3)) * 255)
        self.has_image_been_shown = False


class WinScreenSetter(object):
    def __init__(self, screen_width=1280, screen_height=800):
        self._image_to_show = None
        self.set_screen_res(screen_width=screen_width, screen_height=screen_height)
        self.screen_size = {"width": screen_width, "height": screen_height}
        self.camera = None
        self.top = 0
        self.left = 0
        self.has_image_been_shown = False
        cv2.namedWindow("full_screen", flags=(cv2.WINDOW_NORMAL))
        cv2.setWindowProperty("full_screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.img = None

    def get_frame(self):
        if self.camera is None:
            ret, png = cv2.imencode('.png', self._image_to_show)
        else:
            ret, png = cv2.imencode('.png', self.camera.get_rgb())
        self.has_image_been_shown = True
        return png.tobytes()

    def get_image_to_show(self):
        self.has_image_been_shown = True
        return self._image_to_show

    def set_image(self, img, top=0, left=0):
        # print("set_image_0 %f", time.clock(), file=open("./auto_calibration_log.txt", "a"))
        self._image_to_show = img.copy()
        # self.set_image_top_left(top, left)
        self.top, self.left = top, left
        # print("set_image_1 %f", time.clock(), file=open("./auto_calibration_log.txt", "a"))
        img_full = np.ones((self.screen_size.get("height"), self.screen_size.get("width"), 3)) * 255
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        # print("set_image_2 %f", time.clock(), file=open("./auto_calibration_log.txt", "a"))
        self.img = img

        img_full[top:top + img.shape[0], left:left + img.shape[1], :] = img
        cv2.imshow("full_screen", img_full)
        cv2.waitKey(3)
        # print("set_image_3 %f", time.clock(), file=open("./auto_calibration_log.txt", "a"))
        self.has_image_been_shown = False

    def set_image_top_left(self, top, left):
        if self.top != top or self.left != left:
            self.top, self.left = top, left
            self.set_image(self.img, top, left)

    def set_screen_res(self, screen_width, screen_height):
        self._image_to_show = np.uint8(np.ones((screen_width, screen_height, 3)) * 255)
        self.has_image_been_shown = False
