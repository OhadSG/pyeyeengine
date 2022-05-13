import os
import json

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
ENGINE_PREFERENCES_FILE_PATH = os.getenv("ENGINE_PREFERENCES_FILE_PATH", default="/engineLog/engine_prefs.json")
PACKAGED_PREFS_FILE_PATH = FILE_PATH + "/../utilities/engine_prefs.json"

class EnginePreferences:
    # Constants
    SAVE_DEBUG_FILES = "save_debug_files"
    LOG_LEVEL = "log_level"
    LOG_FOLDER = "log_folder"
    REMOTE_LOGS_ENABLED = "remote_logs_enabled"
    REMOTE_LOGS_URL = "remote_logs_url"
    REMOTE_LOGS_PORT = "remote_logs_port"
    DYNAMIC_SURFACE_DEBUG = "dynamic_surface_debug"
    DYNAMIC_SURFACE_FAR_PLANE = "dynamic_surface_far_plane"
    DYNAMIC_SURFACE_NEAR_PLANE = "dynamic_surface_near_plane"
    DYNAMIC_SURFACE_MIN_FACTOR = "dynamic_surface_min_factor"
    DYNAMIC_SURFACE_CHANGE_THRESHOLD = "dynamic_surface_change_threshold"
    DYNAMIC_SURFACE_MANUAL_MASK_SUPPORT = "dynamic_surface_manual_mask_support"
    X_OFFSET = "x_offset"
    Y_OFFSET = "y_offset"
    SCALE_Y = "y_scale"
    SCALE_X = "x_scale"
    TILT = "tilt"
    FLIP = "flip_data"

    __instance = None

    @staticmethod
    def getInstance() -> 'EnginePreferences':
        if EnginePreferences.__instance is None:
            EnginePreferences.__instance = EnginePreferences()

        return EnginePreferences.__instance

    __SWITCHES = "switches"
    __TEXT_FIELDS = "text_fields"

    def __init__(self):
        self.preferences = {}
        self.load()

    @staticmethod
    def copy_packaged_prefs_file():
        os.system("cp {} /engineLog/".format(PACKAGED_PREFS_FILE_PATH))

    def load(self):
        if not os.path.exists(ENGINE_PREFERENCES_FILE_PATH):
            print("Preferences file not found - copying from engine")
            self.copy_packaged_prefs_file()

        try:
            with open(ENGINE_PREFERENCES_FILE_PATH, "r") as prefs_file:
                self.preferences = json.load(prefs_file)

            with open(PACKAGED_PREFS_FILE_PATH, "r") as packaged_prefs_file:
                packaged_prefs = json.load(packaged_prefs_file)

            packaged_version = int(packaged_prefs["version"])
            if "version" in self.preferences:
                existing_version = int(self.preferences["version"])
            else:
                existing_version = -1

            if existing_version != packaged_version:
                print("Preferences file needs to be updated. existing: {} required: {}".format(existing_version, packaged_version))
                self.copy_packaged_prefs_file()
                self.load()
                return
        except Exception as error:
            print("Error loading preferences file: {}".format(error))
            self.preferences = {}

    def set_switch(self, preference, value):
        try:
            self.preferences[self.__SWITCHES][preference] = value
            with open(ENGINE_PREFERENCES_FILE_PATH, "w") as prefs_file:
                prefs_file.write(json.dumps(self.preferences))
                prefs_file.close()
                EnginePreferences.getInstance().load()
        except Exception as error:
            return False

    def get_switch(self, preference, default=False) -> bool:
        value = get_from_env(preference)
        if value is not None:
            return value == 'true'

        try:
            return self.preferences[self.__SWITCHES][preference]
        except Exception as error:
            return default

    def set_text(self, preference, value):
        try:
            self.preferences[self.__TEXT_FIELDS][preference] = value
            with open(ENGINE_PREFERENCES_FILE_PATH, "w") as prefs_file:
                prefs_file.write(json.dumps(self.preferences))
                prefs_file.close()
                EnginePreferences.getInstance().load()
        except Exception as error:
            return False

    def get_text(self, preference, default="") -> str:
        value = get_from_env(preference)
        if value is not None:
            return value

        try:
            return self.preferences[self.__TEXT_FIELDS][preference]
        except:
            return default

    @property
    def log_folder(self):
        return self.get_text(self.LOG_FOLDER, default='/var/log/pyeyeengine')

    @property
    def remote_logs_host(self):
        return self.get_text(self.REMOTE_LOGS_URL, default='logs.beamforbowl.com')

    @property
    def remote_logs_port(self):
        return int(self.get_text(self.REMOTE_LOGS_PORT, default='5003'))

    @property
    def log_level(self):
        return self.get_text(self.LOG_LEVEL, default='INFO')

    @property
    def logs_file_handler_enabled(self):
        return True

    @property
    def remote_logs_enabled(self):
        return self.get_switch(self.REMOTE_LOGS_ENABLED, default=True)

    @property
    def metrics_enabled(self):
        return True

def get_from_env(preference) -> str:
    return os.getenv('PYEYE_{}'.format(preference.upper()))