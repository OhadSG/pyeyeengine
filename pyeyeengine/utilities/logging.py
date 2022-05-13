import queue
import json

import pytz
import socket as Socket
from datetime import datetime, timezone
from pyeyeengine.utilities.preferences import EnginePreferences
from pyeyeengine.utilities.session_manager import *
import logging
import logging.handlers
from .metrics import Counter

CONNECTION_TIMEOUT = 5
ENABLE_WRITE_TO_FILE = True

logger = logging.getLogger(__name__)

def init_logging(preferences: EnginePreferences = EnginePreferences.getInstance()):
    root_logger = logging.getLogger()
    root_logger.setLevel(preferences.log_level)

    text_formatter = ColoredFormatter(TextFormatter())

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(text_formatter)
    root_logger.addHandler(stderr_handler)

    if preferences.logs_file_handler_enabled:
        versioned_log_folder = os.path.join(preferences.log_folder, Log.get_engine_version())
        os.makedirs(versioned_log_folder, exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(os.path.join(versioned_log_folder, 'log'), when='D')
        file_handler.setFormatter(text_formatter)
        root_logger.addHandler(file_handler)

        file_json_handler = logging.handlers.TimedRotatingFileHandler(os.path.join(preferences.log_folder, 'json-log'), when='D')
        file_json_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_json_handler)

    if preferences.remote_logs_enabled:
        logger.info('Creating remote logger')

        # we create a queue so that the logs are sent in a dedicated thread
        log_queue = queue.Queue()

        # create a handler that sends the logs to the queue
        root_logger.addHandler(logging.handlers.QueueHandler(log_queue))

        # create a thread that will send logs that are in the queue
        queue_listener = logging.handlers.QueueListener(log_queue, RemoteHandler(preferences))
        queue_listener.start()

    root_logger.addHandler(MetricsHandler())


class TextFormatter(logging.Formatter):
    def format(self, record):
        res = "{} - {}   [{}: {}(), {}]: {}".format(
            super().formatTime(record),
            record.levelname,
            record.name,
            record.funcName,
            record.lineno,
            record.msg,
        )

        extra_fields = get_extra_fields(record)
        if len(extra_fields) > 0:
            res += ' ' + json.dumps(extra_fields)

        if record.exc_info is not None:
            res += '\n' + super().formatException(record.exc_info)

        return res


class ColoredFormatter(logging.Formatter):
    """
    Adds color to any formatter.
    """

    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    YELLOW = '\033[0;93m'
    WHITE = '\033[0m'

    COLORS = {
        'WARNING': YELLOW,
        'INFO': BLUE,
        'DEBUG': GREEN,
        'CRITICAL': YELLOW,
        'ERROR': RED
    }

    def __init__(self, inner_formatter: logging.Formatter):
        super().__init__()
        self.inner_formatter = inner_formatter

    def format(self, record):
        inner_res = self.inner_formatter.format(record)
        color = self.COLORS.get(record.levelname)
        if color is None:
            return inner_res
        else:
            return color + inner_res + self.WHITE


log_record_counter = Counter(
    'log_count',
    namespace='pyeye',
)

class MetricsHandler(logging.Handler):
    """
    Creates metrics based on logs.
    """
    def emit(self, record: logging.LogRecord) -> None:
        log_record_counter.inc({
            'level': record.levelname,
        })


class RemoteHandler(logging.Handler):
    logger = logging.getLogger(__name__ + '.KibanaHandler')

    def __init__(self, preferences: EnginePreferences):
        super().__init__()

        self.socket = None
        self.host = preferences.remote_logs_host
        self.port = preferences.remote_logs_port
        self.endpoint = '{}:{}'.format(self.host, self.port)

        if self.endpoint == 'logs.beamforbowl.com:5003':
            logger.info('Using weird remote log formatter')
            self.setFormatter(WeirdRemoteFormatter())
        else:
            logger.info('Using JSON remote log formatter')
            self.setFormatter(JsonFormatter())

    def __create_socket(self):
        logger.info('Connecting to {}'.format(self.endpoint))
        socket = Socket.socket(Socket.AF_INET, Socket.SOCK_STREAM)
        socket.settimeout(CONNECTION_TIMEOUT)
        socket.connect((self.host, self.port))

        RemoteHandler.logger.info("Successfully connected to remote log server {}".format(self.endpoint))

        return socket

    def cleanup(self):
        RemoteHandler.logger.info("RemoteLogger cleanup")
        self.socket.close()

    def emit(self, record: logging.LogRecord):
        if self.socket is None:
            self.socket = self.__create_socket()

        if record.name == self.logger.name:
            # don't emit KibanaHandler's own logs, may cause infinite recursion
            return
        record_text = self.formatter.format(record)
        RemoteHandler.logger.debug("Sending log {}".format(record_text))

        try:
            self.socket.sendall(bytes(record_text + '\n', "utf-8"))
        except:
            RemoteHandler.logger.exception("Error when trying to send log. Endpoint: {}".format(self.endpoint))
            self.socket = None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "level": record.levelname,
            "loggerName": record.name,
            "message": record.msg,
            "lineNumber": record.lineno,
            "function": record.funcName,
            "threadName": record.threadName,
            "created": super().formatTime(record, '%Y-%m-%dT%H:%M:%S%z'),
            "file": record.filename,
            "extra": {
                **get_extra_fields(record),
                'engine_version': Log.get_engine_version(),
                'serial_number': Log.get_system_serial(),
            },
        }
        if record.exc_info is not None:
            obj["stacktrace"] = self.formatException(record.exc_info)
        return json.dumps(obj)


class WeirdRemoteFormatter(logging.Formatter):
    """
    Formats logs in a way that our Logstash/Kibana instance on AWS
    can read.
    It's a pretty weird format, it'd be easier to just use JSON. Maybe someday.
    """
    def format(self, record: logging.LogRecord) -> str:
        # All log fields
        device_time = datetime.now(timezone.utc).isoformat()
        engine_version = Log.get_engine_version()
        serial = Log.get_system_serial()
        platform = " "
        # platform = open("/sys/devices/virtual/android_usb/android0/iManufacturer", "r").read()
        device_raw = '{' + '"platform":"{0}", "serial":"{1}", "engine_version":"{2}"'.format(platform, serial,
                                                                                             engine_version) + '}'
        extra_details_string = None

        extra_details = get_extra_fields(record)

        if len(extra_details) > 0:
            extra_details['session_id'] = "{}".format(get_session_id())
            # CPU Temp: /sys/class/thermal/thermal_zone1/temp

            try:
                extra_details_string = json.dumps(extra_details)
                extra_details_string = extra_details_string.replace("\\n", "8NN8")
                extra_details_string = extra_details_string.replace("\\r", "8NN8")
                extra_details_string = extra_details_string.replace("\\\"", "'")
            except Exception as e:
                print("[ERROR] Could not create JSON: {}".format(e))
        else:
            extra_details_string = json.dumps({"session_id": get_session_id()})

        return "[{}] | {} | {}:{} | {} |".format(
            device_time,
            device_raw,
            record.levelname,
            record.message,
            extra_details_string if extra_details_string is not None else ""
        )


skipped_fields = {
    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename', 'module',
    'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName', 'created', 'msecs',
    'relativeCreated', 'thread', 'threadName', 'processName', 'process',
}

def get_extra_fields(record):
    return {
        key: value
        for (key, value) in record.__dict__.items()
        if key not in skipped_fields
    }


class Log:
    __serial_number = None

    @staticmethod
    def d(msg=None, flow = "standard", extra_details = None):
        if extra_details is None:
            extra_details = {}
        logger.debug(msg, extra={**extra_details, 'flow': flow})

    @staticmethod
    def i(msg=None, flow = "standard", extra_details = None):
        if extra_details is None:
            extra_details = {}
        logger.info(msg, extra={**extra_details, 'flow': flow})

    @staticmethod
    def w(msg=None, flow = "standard", extra_details = None):
        if extra_details is None:
            extra_details = {}
        logger.warning(msg, extra={**extra_details, 'flow': flow})

    @staticmethod
    def e(msg=None, flow = "standard", extra_details = None):
        if extra_details is None:
            extra_details = {}
        logger.error(msg, extra={**extra_details, 'flow': flow})

    ########## Utilities ##########

    @staticmethod
    def get_engine_version() -> str:
        return get_engine_version()

    @staticmethod
    def get_system_serial():
        serial_number = Log.__serial_number
        if serial_number is None:
            try:
                serial_number = get_serial_number()
            except:
                serial_number = 'ERROR'
                Log.__serial_number = serial_number
                logger.exception('Failed to get serial number')

            Log.__serial_number = serial_number

        return serial_number

    @staticmethod
    def get_developer_time():
        local_timezone = pytz.timezone('Asia/Jerusalem')
        format = "%d/%m/%Y %H:%M:%S"
        return datetime.now(local_timezone).strftime(format)

    @staticmethod
    def prepare_logs_for_upload(dest_folder):
        os.system('cp {}/{}/* {}'.format(
            EnginePreferences.getInstance().log_folder,
            Log.get_engine_version(),
            dest_folder,
        ))

def cached(inner_func):
    cached_res = None

    def inner(*args, **kwargs):
        nonlocal cached_res
        if cached_res is None:
            cached_res = inner_func(*args, **kwargs)
        return cached_res

    return inner


def get_serial_number():
    # the serial number is stored in a binary file located at /dev/__properties__/u:object_r:system_prop:s0.
    # running `getprop ro.serialno` in Android's shell reads the serial number from this file.
    # somewhere in the file this appears: serialno211092406301800sys.serialno

    path = os.getenv('PYEYE_SERIAL_NUMBER_PATH', default='/dev/__properties__/u:object_r:system_prop:s0')

    with open(path, 'rb') as system_prop_file:
        system_prop = system_prop_file.read()

    # find the start index
    start_pattern = b'serialno'
    start_pattern_index = system_prop.find(start_pattern)
    if start_pattern_index == -1:
        raise Exception('Failed to find start pattern in system prop file')
    start_serial_index = start_pattern_index + len(start_pattern)

    # find the end index
    end_pattern = b'sys.serialno'
    end_pattern_index = system_prop.find(end_pattern, start_serial_index)
    if end_pattern_index == -1:
        raise Exception('Failed to find end pattern in system prop file')

    # get the serial using the indices
    serial = system_prop[start_serial_index:end_pattern_index]

    # this serial number contains a lot of binary invisible characters
    # so we need to remove them
    serial = ''.join(
        chr(c)
        for c in serial
        if ord('0') <= c <= ord('z')
    )

    if len(serial) != 15:
        raise Exception('Serial number must be 15 characters long, instead it was {}'.format(len(serial)))

    return serial

@cached
def get_engine_version() -> str:
    BASE_PATH = os.path.dirname(os.path.realpath(__file__))
    FULL_VERSION_FILE_NAME = "full_version.txt"
    FULL_VERSION_FILE_PATH = BASE_PATH + "/../" + FULL_VERSION_FILE_NAME

    if os.path.isfile(FULL_VERSION_FILE_PATH):
        with open(FULL_VERSION_FILE_PATH, "r") as file:
            version = file.readline()
            return version
    else:
        return "Unknown"