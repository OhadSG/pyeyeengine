# !/usr/bin/env python
import os
import cv2

import random
import threading
import time
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, render_template, send_file, Response
from flask_socketio import SocketIO

from pyeyeengine.server.screen_setter import ScreenSetter

server_thread = None


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()



FILE_PATH = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)


@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    response.cache_control.public = True
    response.headers['Cache-Control'] = 'no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


def create_chessboard(num_blocks_width=3, num_blocks_height=3, block_size=31):
    return np.tile(np.array([[0, 255], [255, 0]]).repeat(block_size, axis=0).repeat(block_size, axis=1),
                   (num_blocks_height, num_blocks_width))[:-block_size, :-block_size] * 255


def serve_np_image(np_img):
    pil_img = Image.fromarray(np_img)
    # pil_img.show()
    img_io = BytesIO()
    pil_img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


@app.route('/chessboard_image')
def get_chessboard_image():
    return serve_np_image(create_chessboard())


@app.route('/get_image')
def get_screen_setter_image():
    return serve_np_image(screen_setter.get_image_to_show())


@app.route('/')
def get_front_end():
    return open(FILE_PATH + '/android_image_server_side.html', mode="r").read()


def gen(screen_setter):
    while True:
        frame = screen_setter.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    # return Response(gen(screen_setter),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



def run_app():
    try:
        socketio.run(app, port=2222)
    except Exception as exp:
        print(exp)


def disable_logging():
    from flask import Flask
    import logging

    app = Flask(__name__)
    log = logging.getLogger('werkzeug')
    log.disabled = True
    app.logger.disabled = True


socketio = SocketIO(app)
screen_setter = ScreenSetter(socketio)

def start_thread():
    server_thread = threading.Thread(target=run_app, args=(), daemon=True)
    server_thread.start()

def reset_thread():
    if not server_thread is None:
        server_thread.join()
    start_thread()


start_thread()

@socketio.on('message')
def handle_message(message):
    print('received message: ' + message)

if __name__ == '__main__':

    while True:
        time.sleep(.1)
        screen_setter.set_image_top_left(random.randint(0, 200), random.randint(0, 200))
        # socketio.emit('replace_image_data', {'top': random.randint(0, 200), 'left': random.randint(0, 200)})
else:
    pass
    # disable_logging()
