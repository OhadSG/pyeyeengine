#!/usr/bin/env python3

import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import itertools

import requests
import time
import multiprocessing
import cv2
import numpy as np
import socket
import json
import base64
from pyeyeengine.server.socket_wrapper import SocketWrapper
from pyeyeengine.eye_engine.change_detector import ChangeDetector
from pyeyeengine.object_detection.object_detector import Detector
from pyeyeengine.object_detection.object_detector import HandDetector
from pyeyeengine.camera_utils.frame_manager import FrameManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['key-points', 'surface', 'mock', 'rgb', 'change', 'angle', 'depth', 'rgbd'])
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int)
    parser.add_argument('--interval', type=float)
    parser.add_argument('--scale', type=int)
    args = parser.parse_args()

    if args.mode == 'change':
        change_detector = ChangeDetector()
        object_detector = HandDetector()
        for depth_frame in FrameManager.getInstance().depth_stream:
            change_detector.update_background_model(depth_frame, object_detector.get_binary_objects())
            # change_detector.detect_change(depth_frame)
            change_detector.display(depth_frame)

    getter_thread = mode_to_getter[args.mode]

    image_queue = multiprocessing.Queue()

    surface_getter = multiprocessing.Process(
        target=getter_thread,
        args=(
            args.host,
            args.port,
            args.interval,
            args.scale,
            image_queue,
        )
    )

    surface_getter.start()

    try:
        while True:
            image = image_queue.get()
            render_image(image)
    finally:
        surface_getter.kill()


def get_depth(host: str, port: str, interval, scale, queue: multiprocessing.Queue):
    if port is None:
        port = 5006
    if scale is None:
        scale = 1
    if interval is None:
        interval = 0.1

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s = SocketWrapper(s)

    while True:
        start = time.monotonic()
        s.send(json.dumps({
            'name': 'get_depth_map',
        }))
        end = time.monotonic()

        res = json.loads(s.receive_message())
        png_base64 = res.get('data')
        if png_base64 is None:
            print(res)
        else:
            png_bytes = base64.b64decode(png_base64)
            png_numpy = np.frombuffer(png_bytes, dtype=np.uint16)
            png_numpy.shape = (480, 640)
            np.save('table_depth', png_numpy)
            queue.put(resize(png_numpy, scale))

        sleep_until_should_run_again(start, end, interval)


def get_angle(host: str, port: int, interval: float, scale, queue: multiprocessing.Queue):
    if port is None:
        port = 5006
    if interval is None:
        interval = 0.1

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s = SocketWrapper(s)
    sum_x, sum_y, sum_z = 0, 0, 0
    count = 0
    while True:
        start = time.monotonic()
        s.send(json.dumps({
            'name': 'get_offset_angle',
        }))
        end = time.monotonic()

        res = json.loads(s.receive_message())
        angles = res.get('data')
        if angles is None:
            print(res)
        else:
            # print(f"x:{angles['x']}, y:{angles['y']}, z:{angles['z']}")
            data = json.loads(angles)['angles1']

            sum_x += data[0]
            sum_y += data[1]
            sum_z += data[2]
            count += 1
            # queue.put(resize(png_numpy, scale))
            # print(data)
            print([sum_x / count, sum_y / count, sum_z / count])
        sleep_until_should_run_again(start, end, interval)


def key_points_thread(host: str, port: str, interval, scale, queue: multiprocessing.Queue):
    if port is None:
        port = 5006

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s = SocketWrapper(s)

    s.send(json.dumps({
        'name': 'get_key_points',
        'params': {
            'key_points_extractor': 'silhouette'
        }
    }))

    # msg = json.dumps({
    #     'name': 'get_key_points',
    #     'params': {
    #         'key_points_extractor': 'pointing_finger'
    #     }
    # }).encode('utf8')
    # s.send(len(msg).to_bytes(4, byteorder='big', signed=True) + msg)

    while True:
        print('waiting')
        res = json.loads(s.receive_message())
        data = res.get('data')
        if data is None:
            print(res)
            continue

        image = np.zeros((1200, 2000, 3), np.uint8)
        for what_is_this in data:
            for coordinate in what_is_this:
                image = cv2.circle(
                    image,
                    center=(int(coordinate['x']), int(coordinate['y'])),
                    radius=50,
                    color=(255, 255, 255),
                    thickness=-1,
                )

        queue.put(image)
        time.sleep(1)


def rgb_thread(host: str, port: str, interval, scale, queue: multiprocessing.Queue):
    if port is None:
        port = 5006
    if scale is None:
        scale = 1
    if interval is None:
        interval = 0.1

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s = SocketWrapper(s)

    while True:
        start = time.monotonic()
        s.send(json.dumps({
            'name': 'get_rbg_as_png',
        }))
        end = time.monotonic()

        res = json.loads(s.receive_message())
        png_base64 = res.get('data')
        if png_base64 is None:
            print(res)
        else:
            png_bytes = base64.b64decode(png_base64)
            png_numpy = np.frombuffer(png_bytes, dtype=np.uint8)
            image_bytes = cv2.imdecode(png_numpy, flags=cv2.IMREAD_UNCHANGED)

            cv2.imwrite(r'C:\Users\Eyeclick\Desktop\repos\pyeyeengine\pyeyeengine\utilities\local_files\snapshot.png',
                        image_bytes)
            queue.put(resize(image_bytes, scale))

        sleep_until_should_run_again(start, end, interval)


def rgb_over_depth_thread(host: str, port: str, interval, scale, queue: multiprocessing.Queue):
    def zoom_center(img, zoomx, zoomy):
        y_size = img.shape[0]
        x_size = img.shape[1]

        # define new boundaries
        x1 = int(0.5 * x_size * (1 - 1 / zoomx))
        x2 = int(x_size - 0.5 * x_size * (1 - 1 / zoomx))
        y1 = int(0.5 * y_size * (1 - 1 / zoomy))
        y2 = int(y_size - 0.5 * y_size * (1 - 1 / zoomy))

        # first crop image then scale
        img_cropped = img[y1:y2, x1:x2]
        return cv2.resize(img_cropped, None, fx=zoomx, fy=zoomy)

    def shift(mat, dx, dy):  # TODO: not create new - edit mat instead
        new = cv2.copyMakeBorder(mat,
                                 0 if dy < 0 else dy, 0 if dy > 0 else -dy,
                                 0 if dx < 0 else dx, 0 if dx > 0 else -dx,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if dy > 0:
            return new[:-dy, :-dx]
        elif dy == 0:
            return new[:, :-dx]
        else:
            return new[-dy:, :-dx]

    if port is None:
        port = 5006
    if scale is None:
        scale = 1
    if interval is None:
        interval = 0.1

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s = SocketWrapper(s)

    target_interval = interval

    while True:
        start = time.monotonic()
        s.send(json.dumps({
            'name': 'get_depth_raw',
        }))

        res = json.loads(s.receive_message())
        depth_png_base64 = res.get('data')

        start = time.monotonic()
        s.send(json.dumps({
            'name': 'get_rgb_raw',
        }))
        end = time.monotonic()

        res = json.loads(s.receive_message())
        rgb_png_base64 = res.get('data')

        if rgb_png_base64 is None or depth_png_base64 is None:
            print(res)
        else:
            png_bytes = base64.b64decode(depth_png_base64)
            depth = np.frombuffer(png_bytes, dtype=np.uint16)
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
            depth.shape = (480, 640, 3)
            png_bytes = base64.b64decode(rgb_png_base64)
            rgb = np.frombuffer(png_bytes, dtype=np.uint8)
            rgb.shape = (480, 640, 3)

            depth = (depth * (255 / np.max(depth))).astype(np.uint8)
            depth = (depth * 0.6).astype(np.uint8)
            rgb = (rgb * 0.4).astype(np.uint8)

            combined = np.add(depth, rgb)

            queue.put(resize(combined, scale))

        sleep_until_should_run_again(start, end, min(0, target_interval - (time.monotonic() - start)))


def surface_getter_thread(host: str, port: int, interval: float, scale, queue: multiprocessing.Queue):
    if port is None:
        port = 6007
    if scale is None:
        scale = 24
    if interval is None:
        interval = 0

    while True:
        start = time.monotonic()
        dynamic_surface = requests.get('http://{}:{}/get_dynamic_surface_snapshot2'.format(host, port)).json()
        end = time.monotonic()

        print('Response received, took {}'.format(end - start))

        queue.put(resize(dynamic_surface_to_image(dynamic_surface), scale))

        sleep_until_should_run_again(start, end, interval)


def dynamic_surface_to_image(dynamic_surface):
    # map values from (-1, 1) to (0, 255)
    image = [
        ((value + 1) / 2) * 255
        for value in dynamic_surface['values']
    ]

    # turn image into a list of rows
    image = list(
        chunks(image, dynamic_surface['width'])
    )

    # turn image into a numpy array
    image = np.array(image).astype(np.uint8)

    return image


def sleep_until_should_run_again(start, end, interval):
    when_to_run_again = start + interval
    sleep_duration = when_to_run_again - end

    if sleep_duration > 0:
        time.sleep(sleep_duration)


def mock_surface_getter(_host: str, port, interval: float, queue: multiprocessing.Queue):
    values_list = [
        [-1, -1, -1, -1],
        [0, 0, 0, 0],
        [1, 1, 1, 1]
    ]

    for values in itertools.cycle(values_list):
        queue.put({
            'values': values,
            'width': 4,
            'height': 2,
        })

        time.sleep(interval)


mode_to_getter = {
    'mock': mock_surface_getter,
    'surface': surface_getter_thread,
    'key-points': key_points_thread,
    'rgb_plus_d': rgb_thread,
    'angle': get_angle,
    'depth': get_depth,
    'rgbd': rgb_over_depth_thread,
}


def resize(image, scale):
    if scale == 1:
        return image

    width = image.shape[0]
    height = image.shape[1]

    return cv2.resize(
        image,
        (width * scale, height * scale),
        interpolation=cv2.INTER_AREA
    )


def render_image(image):
    cv2.imshow("dynamic surface", image)
    cv2.waitKey(50)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    main()
