import sys
import logging

from pyeyeengine.utilities.rtc_tools import validate_system_time
from pyeyeengine.server.HTTP_Server_Engine import HTTPEngineServer
from pyeyeengine.server.engine_server import EngineServer
from pyeyeengine.engine_installation import dependencies_manager as DM
from pyeyeengine.utilities.preferences import EnginePreferences
from pyeyeengine.server.request_distributor import RequestDistributor
from pyeyeengine.utilities.resource_usage_metrics_reporter import ResourceUsageMetricsReporter
from pyeyeengine.utilities.logging import init_logging
from pyeyeengine.utilities.metrics import init_metrics

from pyeyeengine.utilities.session_manager import *

logger = logging.getLogger(__name__)

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONSOLE_LOG = FILE_PATH + "/../console.txt"
QA_CHECK_FILE_PATH = FILE_PATH + "/../utilities/Scripts/general_scripts/qa_check.py"

def main():
    init_metrics()
    init_logging()
    create_session()

    if os.getenv('PYEYE_DEPENDENCY_CHECK_ENABLED', default='true') == 'true':
        is_installed, missing_packages, description = DM.check_dependencies()
    else:
        is_installed, missing_packages, description = (True, [], '')

    resource_usage_metrics_reporter = ResourceUsageMetricsReporter()
    resource_usage_metrics_reporter.start()

    validate_system_time(should_fix=True)

    if is_installed == True:
        logger.info("Engine dependencies satisfied")
        if os.path.exists(QA_CHECK_FILE_PATH):
            os.system("cp {} ~/".format(QA_CHECK_FILE_PATH))
        logger.info(
            "Engine Preferences",
            extra=EnginePreferences.getInstance().preferences
        )
    else:
        raise DM.DependencyException("Missing required engine dependencies - {}".format(description))

    request_distributor = RequestDistributor()
    HTTPEngineServer(request_distributor).start()
    EngineServer(request_distributor).start()


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
    try:
        main()
    except:
        logger.exception("Engine Terminated")
        exit(1)
