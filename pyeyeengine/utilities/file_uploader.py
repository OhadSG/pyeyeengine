import os
import cv2
import boto3
import threading
from botocore.exceptions import NoCredentialsError
from pyeyeengine.utilities.logging import Log

ENABLED = True
SAVE_LOCALLY = False
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
TEMP_FOLDER_PATH = FILE_PATH + "/temp/"
LOCAL_FOLDER_PATH = FILE_PATH + "/local_files/"

class FileUploader():
    def __init__(self):
        pass

    ACCESS_KEY = 'AKIA5EGXCASQVUR4SYNP'
    SECRET_KEY = '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'

    def upload_image(image, file_name):
        if SAVE_LOCALLY:
            FileUploader.__save_image_locally(image, file_name)
        else:
            if not os.path.exists(TEMP_FOLDER_PATH):
                os.makedirs(TEMP_FOLDER_PATH)

            try:
                FileUploader.save_image(image, file_name)
                FileUploader.upload_in_background(TEMP_FOLDER_PATH + file_name, file_name)
            except Exception as e:
                Log.e("Error uploading image", extra_details={"error": "{}".format(e)})

    def save_image(image, file_name, permanent=False):
        if permanent:
            temp_image_path = LOCAL_FOLDER_PATH + file_name
        else:
            temp_image_path = TEMP_FOLDER_PATH + file_name

        cv2.imwrite(temp_image_path, image)

    def save_image_and_upload(local_file_path, image, file_name):
        if SAVE_LOCALLY:
            FileUploader.__save_image_locally(image, file_name)
        else:
            try:
                cv2.imwrite(local_file_path, image)
                FileUploader.upload_in_background(local_file_path, file_name)
            except Exception as e:
                Log.e("Error uploading image", extra_details={"error": "{}".format(e)})

    def upload_in_background(local_file_path, file_name):
        thread = threading.Thread(target=FileUploader.upload, name="FileUploader", args=(local_file_path, file_name))
        thread.daemon = True
        thread.start()

    def __save_image_locally(image, file_name):
        if image is None:
            Log.d("[ERROR] Image is NONE")
            return

        if os.path.exists(LOCAL_FOLDER_PATH) == False:
            Log.d("Creating folder '{}'".format(LOCAL_FOLDER_PATH))
            os.makedirs(LOCAL_FOLDER_PATH)

        cv2.imwrite(LOCAL_FOLDER_PATH + file_name, image)
        Log.d("{} was saved to {}".format(file_name, LOCAL_FOLDER_PATH))

    def upload(local_file_path, file_name):
        if ENABLED == False:
            return

        bucket = "engine-reports"
        s3 = boto3.client('s3', aws_access_key_id=FileUploader.ACCESS_KEY,
                          aws_secret_access_key=FileUploader.SECRET_KEY)

        remote_file_name = Log.get_system_serial() + "/" + Log.get_engine_version() + "/" + file_name

        try:
            s3.upload_file(local_file_path, bucket, remote_file_name)
            Log.d("Upload successful", flow="FileUploader")
            FileUploader.__delete_temp_file(file_name)
            return True
        except FileNotFoundError:
            Log.w("File not found", flow="FileUploader")
            return False
        except NoCredentialsError:
            Log.w("Credentials not available", flow="FileUploader")
            return False
        except Exception as e:
            Log.w("An unknown error occurred", flow="FileUploader", extra_details={"exception": "{}".format(e)})
            return False

    def __delete_temp_file(file):
        os.system('rm {}'.format(TEMP_FOLDER_PATH + file))

    def __delete_temp():
        os.system('rm -rf {}'.format(TEMP_FOLDER_PATH))

    def clean_local_folder():
        Log.d("Cleaning up {}".format(LOCAL_FOLDER_PATH))
        os.system('rm {}*'.format(LOCAL_FOLDER_PATH))