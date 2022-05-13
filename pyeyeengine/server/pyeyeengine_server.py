import os
import sys
import traceback

from pyeyeengine.server.engine_server import EngineServer
from pyeyeengine.utilities.logging import Log
from pyeyeengine.engine_installation import dependencies_manager as DM
from pyeyeengine.camera_utils.frame_manager import FrameManager

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONSOLE_LOG = FILE_PATH + "/../console.txt"
QA_CHECK_FILE_PATH = FILE_PATH + "/../utilities/Scripts/general_scripts/qa_check.py"

def main():
    is_installed, missing_packages, description = DM.check_dependencies()

    if is_installed == True:
        Log.d("Engine dependencies satisfied")
        if os.path.exists(QA_CHECK_FILE_PATH):
            os.system("cp {} ~/".format(QA_CHECK_FILE_PATH))
        EngineServer().start()
    else:
        Log.e("Missing required engine dependencies", extra_details=missing_packages)
        raise DM.DependencyException("Missing required engine dependencies - {}".format(description))


def remove_old_console_logs():
    if os.path.exists(PATH_TO_CONSOLE_LOG) and os.path.getsize(PATH_TO_CONSOLE_LOG) > 100000:
        console_log_text = open(PATH_TO_CONSOLE_LOG, "r").read()
        os.remove(PATH_TO_CONSOLE_LOG)
        open(PATH_TO_CONSOLE_LOG, "w").write(console_log_text[int(len(console_log_text) / 2):])


def start_console_log():
    global orig_stdout, f
    orig_stdout = sys.stdout
    f = open(PATH_TO_CONSOLE_LOG, 'a')
    sys.stdout = f


def close_console_log():
    sys.stdout = orig_stdout
    f.close()

if __name__ == '__main__':
    # remove_old_console_logs()
    # start_console_log()

    try:
        main()
    except Exception as e:
        Log.e("Engine Terminated", extra_details={"exception": "{}".format(e), "stacktrace":traceback.format_exc()})

    # close_console_log()
