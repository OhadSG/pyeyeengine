import glob
import json
import socket
import time
import cv2
from threading import Thread
from pyeyeengine.server.socket_wrapper import SocketWrapper
# from pyeyeengine.server import pyeyeengine_server as server
#from pyeyeengine.camera_utils.frame_manager import FrameManager
from pyeyeengine.utilities.file_uploader import FileUploader
import pyeyeengine.utilities.global_params as Globals

def open_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 5006))
    MESSAGE_SIZE_INDICATOR_LENGTH = 4
    return SocketWrapper(sock)

def test_unmount_usb(socket):
    print("Will test unmounting/remounting USB devices")
    socket.send(json.dumps({'name': "test_usb"}))

def test_framemanager_reset(socket):
    print("Trying to reset FrameManager")
    socket.send(json.dumps({'name': "reset_frame_manager"}))

def get_current_depth_frame(socket):
    print("Requesting depth rame")
    socket.send(json.dumps({'name': 'get_depth_frame'}))

def record_streams(socket):
    print("Recording streams")
    socket.send(json.dumps({'name': 'record', 'params': {'record': 1}}))
    time.sleep(10)
    print("Finished recording")
    socket.send(json.dumps({'name': 'record', 'params': {'record': 0}}))

if __name__ == '__main__':
    # thread = Thread(target=server.main)
    # thread.start()
    #
    # time.sleep(20)
    #
    socket_wrapper = open_socket()
    #test_unmount_usb(socket_wrapper)
    # test_framemanager_reset(socket_wrapper)
    # get_current_depth_frame(socket_wrapper)
    record_streams(socket_wrapper)
    response = socket_wrapper.receive_message()
    print("Response: {}".format(response))

    # while (True):
    #     message = sw.receive_message()


# import os
# import argparse
# import subprocess
#
# path='/sys/bus/usb/devices/'
#
# def runbash(cmd):
#     p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
#     out = p.stdout.read().strip()
#     return out
#
# def reset_device(dev_num):
#     sub_dirs = []
#     for root, dirs, files in os.walk(path):
#             for name in dirs:
#                     sub_dirs.append(os.path.join(root, name))
#
#     dev_found = 0
#     for sub_dir in sub_dirs:
#             if True == os.path.isfile(sub_dir+'/devnum'):
#                     fd = open(sub_dir+'/devnum','r')
#                     line = fd.readline()
#                     if int(dev_num) == int(line):
#                             print ('Your device is at: '+sub_dir)
#                             dev_found = 1
#                             break
#
#                     fd.close()
#
#     if dev_found == 1:
#             reset_file = sub_dir+'/authorized'
#             runbash('echo 0 > '+reset_file)
#             runbash('echo 1 > '+reset_file)
#             print ('Device reset successful')
#
#     else:
#             print ("No such device")
#
# def main():
#     reset_device(9)
#
# if __name__=='__main__':
#     main()