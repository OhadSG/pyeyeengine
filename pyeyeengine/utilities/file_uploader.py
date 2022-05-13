import os
import cv2
import boto3
import threading
import shutil
from botocore.exceptions import NoCredentialsError
from pyeyeengine.utilities.logging import Log
from pyeyeengine.utilities.network_check import can_access_internet
from pyeyeengine.utilities.preferences import EnginePreferences

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
TEMP_FOLDER_PATH = FILE_PATH + "/temp/"
LOCAL_FOLDER_PATH = FILE_PATH + "/local_files/"
ARCHIVE_TEMP_FOLDER = TEMP_FOLDER_PATH + "archives/"

class FileUploader:
    ACCESS_KEY = 'AKIA5EGXCASQVUR4SYNP'
    SECRET_KEY = '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'

    @staticmethod
    def __archive_folder(path, archive_name):
        if not os.path.exists(ARCHIVE_TEMP_FOLDER):
            os.makedirs(ARCHIVE_TEMP_FOLDER, True)

        shutil.make_archive(ARCHIVE_TEMP_FOLDER + archive_name, 'zip', path)

        return ARCHIVE_TEMP_FOLDER + archive_name + ".zip"

    @staticmethod
    def __save_text_to_file_locally(text, file_name, folder):
        if text is None:
            Log.d("[ERROR] Text is NONE")
            return

        save_path = LOCAL_FOLDER_PATH

        if os.path.exists(LOCAL_FOLDER_PATH) == False:
            Log.d("Creating folder '{}'".format(LOCAL_FOLDER_PATH))
            os.makedirs(LOCAL_FOLDER_PATH)

        if folder is not None:
            save_path = LOCAL_FOLDER_PATH + folder

            if os.path.exists(LOCAL_FOLDER_PATH + folder) == False:
                Log.d("Creating folder '{}'".format(LOCAL_FOLDER_PATH + folder))
                os.makedirs(LOCAL_FOLDER_PATH + folder)

        with open(save_path + file_name, "w+") as text_file:
            text_file.write(text)
            text_file.close()
            Log.d("{} was saved to {}".format(file_name, save_path))

    @staticmethod
    def __save_image_locally(image, file_name, folder):
        if image is None:
            Log.d("[ERROR] Image is NONE")
            return

        save_path = LOCAL_FOLDER_PATH

        if os.path.exists(LOCAL_FOLDER_PATH) == False:
            Log.d("Creating folder '{}'".format(LOCAL_FOLDER_PATH))
            os.makedirs(LOCAL_FOLDER_PATH)

        if folder is not None:
            save_path = LOCAL_FOLDER_PATH + folder

            if os.path.exists(LOCAL_FOLDER_PATH + folder) == False:
                Log.d("Creating folder '{}'".format(LOCAL_FOLDER_PATH + folder))
                os.makedirs(LOCAL_FOLDER_PATH + folder)

        cv2.imwrite(save_path + file_name, image)
        Log.d("{} was saved to {}".format(file_name, save_path))

    @staticmethod
    def __upload_in_background(local_file_path, file_name):
        thread = threading.Thread(target=FileUploader.__upload, name="FileUploader", args=(local_file_path, file_name))
        thread.daemon = True
        thread.start()

    @staticmethod
    def __upload(local_file_path, file_name):
        if can_access_internet() is False:
            Log.e("Not connected to internet, cannot upload file", flow="FileUploader")
            return False

        bucket = "engine-reports"
        s3 = boto3.client('s3', aws_access_key_id=FileUploader.ACCESS_KEY,
                          aws_secret_access_key=FileUploader.SECRET_KEY)

        remote_file_name = Log.get_system_serial() + "/" + Log.get_engine_version() + "/" + file_name

        Log.d("Uploading file: {}".format(remote_file_name))

        try:
            s3.upload_file(local_file_path, bucket, remote_file_name)
            Log.d("Upload successful", flow="FileUploader")
            return True
        except FileNotFoundError:
            Log.w("File not found: {}".format(local_file_path), flow="FileUploader")
            return False
        except NoCredentialsError:
            Log.w("Credentials not available", flow="FileUploader")
            return False
        except TimeoutError:
            Log.w("Connection to S3 has timed out", flow="FileUploader")
            return False
        except Exception as e:
            Log.w("An unknown error occurred", flow="FileUploader", extra_details={"exception": "{}".format(e)})
            return False

    ###### Public Functions ######

    @staticmethod
    def clean_local_folder():
        os.system('rm {}*'.format(LOCAL_FOLDER_PATH))
        os.system('rm -rf {}*'.format(ARCHIVE_TEMP_FOLDER))

    @staticmethod
    def save_text_file(text, file_name, folder=None):
        if EnginePreferences.getInstance().get_switch(EnginePreferences.SAVE_DEBUG_FILES):
            FileUploader.__save_text_to_file_locally(text, file_name, folder)

    @staticmethod
    def save_image(image, file_name, folder=None):
        if EnginePreferences.getInstance().get_switch(EnginePreferences.SAVE_DEBUG_FILES):
            FileUploader.__save_image_locally(image, file_name, folder)

    @staticmethod
    def upload_folder(path, archive_name="Untitled"):
        archive_path = FileUploader.__archive_folder(path, archive_name)
        FileUploader.__upload_in_background(archive_path, archive_name + ".zip")

    @staticmethod
    def clear_archives_folder():
        os.system('rm -rf {}'.format(ARCHIVE_TEMP_FOLDER))