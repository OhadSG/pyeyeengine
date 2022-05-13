import numpy as np
import cv2
from primesense import openni2  # , nite2
from primesense import _openni2 as c_api
from primesense.openni2 import IMAGE_REGISTRATION_DEPTH_TO_COLOR
import platform
import inspect
import os
from pyeyeengine.utilities.init_openni import init_openni

def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

class DepthGetter():
    def __init__(self):
        np.set_printoptions(threshold=np.inf)
        self.device = None
        self.depth_stream = None
        self.rgb_stream = None
        self.color_index = 16
        self.color_maps = []
        self.control_image = None

        self.load_driver()
        self.load_color_maps()

    def load_driver(self):
        init_openni()

        if openni2.is_initialized():
            print("openNI2 initialized")

        ## Register the device
        self.device = openni2.Device.open_any()

        ## Create the streams
        self.depth_stream = self.device.create_depth_stream()
        self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM,
                                                       resolutionX=640,
                                                       resolutionY=480,
                                                       fps=30))
        self.depth_stream.start()
        self.depth_stream.set_mirroring_enabled(False)

        self.rgb_stream = self.device.create_color_stream()
        self.rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                     resolutionX=640,
                                                     resolutionY=480,
                                                     fps=30))
        self.rgb_stream.start()
        self.rgb_stream.set_mirroring_enabled(False)

        self.device.set_image_registration_mode(IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    def load_color_maps(self):
        self.colormaps = [cv2.COLORMAP_AUTUMN,
                 cv2.COLORMAP_BONE,
                 cv2.COLORMAP_CIVIDIS,
                 cv2.COLORMAP_COOL,
                 cv2.COLORMAP_HOT,
                 cv2.COLORMAP_HSV,
                 cv2.COLORMAP_INFERNO,
                 cv2.COLORMAP_JET,
                 cv2.COLORMAP_MAGMA,
                 cv2.COLORMAP_OCEAN,
                 cv2.COLORMAP_PARULA,
                 cv2.COLORMAP_PINK,
                 cv2.COLORMAP_PLASMA,
                 cv2.COLORMAP_RAINBOW,
                 cv2.COLORMAP_SPRING,
                 cv2.COLORMAP_SUMMER,
                 cv2.COLORMAP_TURBO,
                 cv2.COLORMAP_TWILIGHT,
                 cv2.COLORMAP_TWILIGHT_SHIFTED,
                 cv2.COLORMAP_VIRIDIS,
                 cv2.COLORMAP_WINTER]

    def process_click(event, x, y, flags, params, color_index):
        # check if the click is within the dimensions of the button
        if event == cv2.EVENT_LBUTTONDOWN:
            if y > button[0] and y < button[1] and x > button[2] and x < button[3]:
                color_index -= 1
                color_index = max(0, min(color_index, len(colormaps) - 1))
            elif y > button2[0] and y < button2[1] and x > button2[2] and x < button2[3]:
                color_index += 1
                color_index = max(0, min(color_index, len(colormaps) - 1))

    def prepare_to_show(self):
        # [ymin, ymax, xmin, xmax]
        button = [20, 60, 20, 220]
        button2 = [20, 60, 240, 440]

        cv2.namedWindow('Control')
        cv2.setMouseCallback('Control', self.process_click, param=self.color_index)

        self.control_image = np.zeros((80, 460), np.uint8)
        self.control_image[button[0]:button[1], button[2]:button[3]] = 180
        cv2.putText(self.control_image, 'Previous', (button[2] + 10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)

        self.control_image[button2[0]:button2[1], button2[2]:button2[3]] = 255
        cv2.putText(self.control_image, 'Next', (button2[2] + 10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)

    def get_depth_frame(self):
        depth_frame = self.depth_stream.read_frame()
        buffer = depth_frame.get_buffer_as_uint16()
        depth_array = np.fromstring(buffer, dtype=np.uint16)
        depth_map = depth_array.reshape(480, 640)
        max_depth_map = np.max(depth_map)
        min_depth_map = np.min(depth_map)
        depth_map_adj = np.uint8(((depth_map - min_depth_map) / (np.max((max_depth_map - min_depth_map, 1)))) * 255)
        return depth_map_adj

    def show(self):
        depth_frame = self.get_depth_frame()
        depth = cv2.applyColorMap(depth_frame, self.colormaps[self.color_index])
        # depth = cv2.resize(depth, (640, 480))

        bgr = np.fromstring(self.rgb_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint8).reshape(
            480, 640, 3)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        stack = np.hstack((depth, rgb))
        cv2.putText(stack, 'Selected: {}'.format(self.color_index), (580, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.imshow('Control', self.control_image)
        cv2.imshow('Camera Feed', stack)

        if cv2.waitKey(1) == ord('a'):
            self.color_index += 1
            print(self.color_index)
            self.color_index = max(0, min(self.color_index, len(self.colormaps) - 1))
        elif cv2.waitKey(1) == ord('s'):
            self.color_index -= 1
            print(self.color_index)
            self.color_index = max(0, min(self.color_index, len(self.colormaps) - 1))

# if __name__ == '__main__':
#     getter = DepthGetter()
#     getter.prepare_to_show()
#
#     while True:
#         frame = getter.get_depth_frame()
#         cv2.imshow("test", frame)
#         cv2.waitKey(1)

