import inspect
import os
import socket as Socket
from datetime import date, datetime, timezone
import json
import sys
import pytz

ENABLED = True
VERBOSE = False
ENABLED_SEND_TO_REMOTE = True
ENABLE_WRITE_TO_FILE = True
ENABLE_DEBUG_LOGS = True
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
LOG_FOLDER_PATH = BASE_PATH + "/logs/"

GREEN = '\033[0;32m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
PURPLE = '\033[0;35m'
YELLOW = '\033[0;93m'
WHITE = '\033[0m'

REMOTE_LOGS_URL = "logs.beamforbowl.com"
REMOTE_LOGS_PORT = 5003

class Log():
    # DEBUG
    def d(msg=None, flow = "standard", extra_details = None):
        if VERBOSE:
            Log.send_to_remote("DEBUG", flow, msg if msg is not None else '', extra_details=extra_details)

        if ENABLED and ENABLE_DEBUG_LOGS:
            (filename, function, line) = Log.get_frame_info()
            log_text = "{}DEBUG   [{}: {}(), {}]: {}\033[0m".format(GREEN, filename, function, line,
                                                                    msg if msg is not None else '')

            if extra_details is not None:
                log_text = log_text + " [Details: {}]".format(json.dumps(extra_details))

            print(log_text)

            if ENABLE_WRITE_TO_FILE:
                Log.write_to_file(log_text)

    # INFO
    def i(msg=None, flow = "standard", extra_details = None):
        Log.send_to_remote("INFO", flow, msg if msg is not None else '', extra_details=extra_details)
        if ENABLED:
            (filename, function, line) = Log.get_frame_info()
            log_text = "{}INFO    [{}: {}(), {}]: {}\033[0m".format(BLUE, filename, function, line,
                                                                    msg if msg is not None else '')

            if extra_details is not None:
                log_text = log_text + " [Details: {}]".format(json.dumps(extra_details))

            print(log_text)

            if ENABLE_WRITE_TO_FILE:
                Log.write_to_file(log_text)

    # WARNING
    def w(msg=None, flow = "standard", extra_details = None):
        Log.send_to_remote("WARNING", flow, msg if msg is not None else '', extra_details=extra_details)
        if ENABLED:
            (filename, function, line) = Log.get_frame_info()
            log_text = "{}WARNING [{}: {}(), {}]: {}\033[0m".format(YELLOW, filename, function, line,
                                                                    msg if msg is not None else '')

            if extra_details is not None:
                log_text = log_text + " [Details: {}]".format(json.dumps(extra_details))

            print(log_text)

            if ENABLE_WRITE_TO_FILE:
                Log.write_to_file(log_text)

    # ERROR
    def e(msg=None, flow = "standard", extra_details = None):
        Log.send_to_remote("ERROR", flow, msg if msg is not None else '', extra_details=extra_details)
        if ENABLED:
            (filename, function, line) = Log.get_frame_info()
            log_text = "{}ERROR   [{}: {}(), {}]: {}\033[0m".format(RED, filename, function, line,
                                                                    msg if msg is not None else '')

            if extra_details is not None:
                log_text = log_text + " [Details: {}]".format(json.dumps(extra_details))

            print(log_text)

            if ENABLE_WRITE_TO_FILE:
                Log.write_to_file(log_text)

    # TODO
    def t(msg=None):
        Log.send_to_remote("TODO", msg if msg is not None else '')
        if ENABLED:
            (filename, function, line) = Log.get_frame_info()
            log_text = "{}TODO    [{}: {}(), {}]: {}\033[0m".format(PURPLE, filename, function, line,
                                                                    msg if msg is not None else '')
            print(log_text)

            if ENABLE_WRITE_TO_FILE:
                Log.write_to_file(log_text)

    # CALLER
    def caller():
        if ENABLED:
            (filename, function, line) = Log.get_frame_info()
            (filename1, function1, line1) = Log.get_frame_info(3)
            print("{}CALLER  [{}: {}(), {}]: {}\033[0m".format(GREEN, filename, function, line,
                                                               "Called from [{}: {}(), {}]".format(filename1, function1, line1)))

    def get_engine_version():
        BASE_PATH = os.path.dirname(os.path.realpath(__file__))
        FULL_VERSION_FILE_NAME = "full_version.txt"
        FULL_VERSION_FILE_PATH = BASE_PATH + "/../" + FULL_VERSION_FILE_NAME

        if os.path.isfile(FULL_VERSION_FILE_PATH):
            with open(FULL_VERSION_FILE_PATH, "r") as file:
                version = file.readline()
                return version
        else:
            return "1.0.7.1111"

    def get_frame_info(frames_back=2):
        if ENABLED == False:
            return

        try:
            if inspect.stack(0)[frames_back] == None:
                return "Unknown", "Log", "Origin"
            else:
                frame = inspect.stack(0)[frames_back]
                function = frame.function
                line = inspect.getlineno(frame[0])
                filename = os.path.basename(frame.filename)

            return filename, function, line
        except:
            return "Unknown", "Log", "Origin"

    # def write_to_file(log_text):
    #     if True:
    #         Log.send_to_remote(log_text)
    #
    #     if os.path.exists(LOG_FOLDER_PATH + "log.txt"):
    #         outFile = open(LOG_FOLDER_PATH + "log.txt", "a+")
    #     else:
    #         outFile = open(LOG_FOLDER_PATH + "log.txt", "w+")
    #
    #     outFile.write(log_text)
    #     outFile.write("\n")
    #     outFile.close()

    def get_system_serial():
        return open("/sys/devices/virtual/android_usb/android0/iSerial", "r").read()

    def write_to_file(log_text):
        # today = str(date.today()).replace("-", "")
        today = str(date.today())
        log_file_location = "/root/engine_logs/{}".format(Log.get_engine_version())
        log_file_name = "{}/{}.txt".format(log_file_location, today)
        os.system("mkdir -p /root/engine_logs")
        os.system("mkdir -p {}".format(log_file_location))

        output_file = open("{}".format(log_file_name), "a+")
        output_file.write("{} {}".format(Log.get_developer_time(), log_text))
        output_file.write("\n")
        output_file.close()

    def get_developer_time():
        local_timezone = pytz.timezone('Asia/Jerusalem')
        format = "%d/%m/%Y %H:%M:%S"
        return datetime.now(local_timezone).strftime(format)

    def send_to_remote(log_level, flow="standard", message="", extra_details=None):
        if ENABLED_SEND_TO_REMOTE == False:
            return

        # All log fields
        device_time = datetime.now(timezone.utc).isoformat()
        engine_version = Log.get_engine_version()
        serial = Log.get_system_serial()
        platform = open("/sys/devices/virtual/android_usb/android0/iManufacturer", "r").read()
        device_raw = '{' +'"platform":"{0}", "serial":"{1}", "engine_version":"{2}"'.format(platform, serial, engine_version) + '}'
        extra_details_string = None

        if extra_details is not None:
            # if type(extra_details) is dict:
            extra_details['flow'] = "{}".format(flow)
            # CPU Temp: /sys/class/thermal/thermal_zone1/temp

            try:
                extra_details_string = json.dumps(extra_details)
                extra_details_string = extra_details_string.replace("\\n", "8NN8")
                extra_details_string = extra_details_string.replace("\\r", "8NN8")
                extra_details_string = extra_details_string.replace("\\\"", "'")
            except Exception as e:
                print("[ERROR] Could not create JSON: {}".format(e))
        else:
            extra_details_string = json.dumps({"flow": flow})

        text_to_send = "[{}] | {} | {}:{} | {} |".format(device_time, device_raw, log_level, message, extra_details_string if extra_details_string is not None else "")

        try:
            socket = Socket.socket(Socket.AF_INET, Socket.SOCK_STREAM)
            socket.connect((REMOTE_LOGS_URL, REMOTE_LOGS_PORT))
            socket.sendall(bytes(text_to_send, "utf-8"))
            socket.close()
        except Exception as e:
            print("[ERROR] Error when trying to send log: {} [{}, {}]".format(e, REMOTE_LOGS_URL, REMOTE_LOGS_PORT))