import os
import sys
import json
import pytz
import threading
from datetime import date, datetime, timezone
from pyeyeengine.utilities.session_manager import *

ENABLE_WRITE_TO_FILE = True

def __get_engine_version():
    BASE_PATH = os.path.dirname(os.path.realpath(__file__))
    FULL_VERSION_FILE_NAME = "full_version.txt"
    FULL_VERSION_FILE_PATH = BASE_PATH + "/../" + FULL_VERSION_FILE_NAME

    if os.path.isfile(FULL_VERSION_FILE_PATH):
        with open(FULL_VERSION_FILE_PATH, "r") as file:
            version = file.readline()
            return version
    else:
        return "Unknown"

def __get_developer_time():
    local_timezone = pytz.timezone('Asia/Jerusalem')
    format = "%d/%m/%Y %H:%M:%S"
    return datetime.now(local_timezone).strftime(format)

def write_to_file(folder_path, log_text, extra_details=None):
    assert folder_path is not None, "File path must be provided!"

    if extra_details is not None:
        log_text = log_text + " {}".format(json.dumps(extra_details))

    if ENABLE_WRITE_TO_FILE:
        __handle_write(folder_path, log_text, True)


def __handle_write(folder_path, log_text, in_background=False):
    if in_background:
        writer_thread = threading.Thread(target=__write, name="FileWriter", args=(folder_path, log_text,))
        writer_thread.daemon = True
        writer_thread.start()
    else:
        __write(folder_path, log_text)


def __write(log_folder_path, log_text):
    today = str(date.today())
    log_file_location = "{}/{}".format(log_folder_path, __get_engine_version())
    log_file_name = "{}/{}.txt".format(log_file_location, today)
    os.system("mkdir -p {}".format(log_folder_path))
    os.system("mkdir -p {}".format(log_file_location))

    output_file = open("{}".format(log_file_name), "a+")
    output_file.write("{} {} {}".format(get_session_id(), __get_developer_time(), log_text))
    output_file.write("\n")
    output_file.close()