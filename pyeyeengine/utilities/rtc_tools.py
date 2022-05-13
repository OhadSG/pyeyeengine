import os
import subprocess as sp
import time
from datetime import date, datetime, timezone
from pyeyeengine.utilities.network_check import can_access_internet
from pyeyeengine.utilities.logging import Log
from pyeyeengine.utilities.file_writer import write_to_file

# Constants
RTC_I2C_ADDRESS = 0x68
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
LATEST_TIME_FILENAME = "/latest_time.txt"
FLOW = "system_time"
NETWORK_TIME_ATTEMPTS = 5
RTC_READ_ATTEMPTS = 5
REPORTS_FOLDER = os.getenv("REPORTS_FOLDER", default="/root/rtc_reports")

# Implementation

def get_network_time():
    retry_attempts = 0

    while retry_attempts < NETWORK_TIME_ATTEMPTS:
        retry_attempts += 1
        Log.i("Trying to get network time", extra_details={"attempt": "{}".format(retry_attempts)}, flow=FLOW)
        write_to_file(REPORTS_FOLDER, "Trying to get network time", extra_details={"attempt": "{}".format(retry_attempts)})
        p = sp.Popen(['nc', 'time.nist.gov', '13'], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        output, catch = p.communicate(b"input data that is passed to subprocess' stdin")
        result_string = output.decode("utf-8")
        caught_error = catch.decode("utf-8")

        if result_string is not None and len(result_string) > 1:
            try:
                result_array = result_string.split(" ")
                combined_time = "{}-{}".format(result_array[1], result_array[2])
                date_time_obj = datetime.strptime(combined_time, '%y-%m-%d-%H:%M:%S')
                os.system("date +%Y%m%d -s {}".format(date_time_obj.strftime('%Y%m%d')))
                os.system("date +%T -s '{}'".format(date_time_obj.strftime('%H:%M:%S')))
                return date_time_obj
            except Exception as e:
                return None
        elif caught_error is not None and len(caught_error) > 1:
            Log.e("Error getting network time", extra_details={"error": caught_error}, flow=FLOW)
            write_to_file(REPORTS_FOLDER, "Error getting network time", extra_details={"error": caught_error})
        else:
            Log.e("Unknown error while getting network time", flow=FLOW)
            write_to_file(REPORTS_FOLDER, "Unknown error while getting network time")

    return None

def rtc_installed():
    return False
    p = sp.Popen(['i2cget', '-y', '-f', '1', '{}'.format(RTC_I2C_ADDRESS)], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    output, catch = p.communicate(b"input data that is passed to subprocess' stdin")

    if "Error: Read failed" in catch.decode("utf-8"):
        write_to_file(REPORTS_FOLDER, "RTC not found - read failed")
        return False
    elif "resource busy" in catch.decode("utf-8"):
        write_to_file(REPORTS_FOLDER, "RTC not found - resource busy")
        return False
    else:
        write_to_file(REPORTS_FOLDER, "RTC found")
        return True

def save_rtc_time(custom=None):
    try:
        if custom is not None:
            today = custom
        else:
            today = datetime.today().strftime('%y-%m-%d-%H-%M-%S')

        date_array = today.split("-")
        os.system("i2cset -y -f 1 {} {} {}".format(RTC_I2C_ADDRESS, 0, hex(int(date_array[5]))))
        os.system("i2cset -y -f 1 {} {} {}".format(RTC_I2C_ADDRESS, 1, hex(int(date_array[4]))))
        os.system("i2cset -y -f 1 {} {} {}".format(RTC_I2C_ADDRESS, 2, hex(int(date_array[3]))))
        os.system("i2cset -y -f 1 {} {} {}".format(RTC_I2C_ADDRESS, 4, hex(int(date_array[2]))))
        os.system("i2cset -y -f 1 {} {} {}".format(RTC_I2C_ADDRESS, 5, hex(int(date_array[1]))))
        os.system("i2cset -y -f 1 {} {} {}".format(RTC_I2C_ADDRESS, 6, hex(int(date_array[0]))))
        Log.i("Saved time to RTC", extra_details={"saved_time": "{}".format(today)}, flow=FLOW)
        write_to_file(REPORTS_FOLDER, "Saved time to RTC", extra_details={"saved_time": "{}".format(today)})
        return True
    except Exception as e:
        Log.e("Could not save time to RTC", extra_details={"error": "{}".format(e)}, flow=FLOW)
        write_to_file(REPORTS_FOLDER, "Could not save time to RTC", extra_details={"error": "{}".format(e)})
        return False

def get_rtc_time():
    read_attempts = 0

    while read_attempts < RTC_READ_ATTEMPTS:
        date = []
        os.system("i2cset -y -f 1 {} 0x0".format(RTC_I2C_ADDRESS))

        for i in range(8):
            p = sp.Popen(['i2cget', '-y', '-f', '1', '{}'.format(RTC_I2C_ADDRESS)], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
            output, catch = p.communicate(b"input data that is passed to subprocess' stdin")
            date.append(int(output.decode("utf-8"), 16))

        read_date_string = '{}-{}-{}-{}-{}-{}'.format(date[6], date[5], date[4], date[2], date[1], date[0] if date[0] < 60 else 0)

        try:
            return datetime.strptime(read_date_string, '%y-%m-%d-%H-%M-%S')
        except Exception as e:
            Log.e("Error reading RTC time", extra_details={"error": "{}".format(e)}, flow=FLOW)
            write_to_file(REPORTS_FOLDER, "Error reading RTC time", extra_details={"error": "{}".format(e)})

            if "does not match format" in str(e):
                read_attempts += 1
                write_to_file(REPORTS_FOLDER, "Will attempt to read RTC time again", extra_details={"attempt": read_attempts})
            else:
                write_to_file(REPORTS_FOLDER, "Could not get a proper date")
                return datetime.strptime("70-01-01-00-00-00", '%y-%m-%d-%H-%M-%S')

    write_to_file(REPORTS_FOLDER, "Exceeded attempts to get date")
    return datetime.strptime("70-01-01-00-00-00", '%y-%m-%d-%H-%M-%S')

def is_rtc_time_valid():
    if not rtc_installed(): return False
    rtc_time = get_rtc_time()
    return int(rtc_time.year) >= 2021

def fix_time_without_rtc(should_fix=False):
    # TODO: Remove!
    if can_access_internet():
        with open(BASE_PATH + LATEST_TIME_FILENAME, "w+") as file:
            current_date = datetime.now(timezone.utc).isoformat()
            file.write(current_date)
    else:
        if should_fix:
            current_date = datetime.now(timezone.utc).isoformat()

            if int(date.today().year) < 2021 and os.path.isfile(BASE_PATH + LATEST_TIME_FILENAME):
                with open(BASE_PATH + LATEST_TIME_FILENAME, 'r') as file:
                    saved_date = file.read()
                    os.system('date -s {}'.format(saved_date))
            else:
                os.system('date -s 01/01/2021')

            current_date = datetime.now(timezone.utc).isoformat()

def validate_system_time(should_fix=False):
    Log.i("Validating system time", flow=FLOW)
    write_to_file(REPORTS_FOLDER, "Validating system time")

    if rtc_installed():
        today = datetime.today()
        today_string = today.strftime('%y-%m-%d-%H-%M-%S')

        if not can_access_internet() and int(today.year) <= 2020:
            Log.e("System time is wrong", extra_details={"system_time": "{}".format(today_string)}, flow=FLOW)
            write_to_file(REPORTS_FOLDER, "System time is wrong", extra_details={"system_time": "{}".format(today_string)})

            network_time = get_network_time()

            if network_time:
                Log.i("Fixed wrong system time", extra_details={"wrong_time": today_string, "network_time": network_time.strftime('%y-%m-%d-%H-%M-%S')}, flow=FLOW)
                write_to_file(REPORTS_FOLDER, "Fixed wrong system time", extra_details={"wrong_time": today_string, "network_time": network_time.strftime('%y-%m-%d-%H-%M-%S')})
                save_rtc_time()
            else:
                if is_rtc_time_valid():
                    rtc_time = get_rtc_time()
                    Log.i("RTC time valid, updating system", extra_details={"rtc_time": "{}".format(rtc_time)}, flow=FLOW)
                    write_to_file(REPORTS_FOLDER, "RTC time valid, updating system", extra_details={"rtc_time": "{}".format(rtc_time)})
                    os.system("date +%Y%m%d -s {}".format(rtc_time.strftime('%Y%m%d')))
                    os.system("date +%T -s '{}'".format(rtc_time.strftime('%H:%M:%S')))
                else:
                    Log.e("RTC time invalid, setting default", flow=FLOW)
                    write_to_file(REPORTS_FOLDER, "RTC time invalid, setting default")
                    os.system("date -s 06/01/2021")
                    save_rtc_time()

                if can_access_internet():
                    get_network_time()
                    save_rtc_time()
                else:
                    today = datetime.today()
                    today_string = today.strftime('%y-%m-%d-%H-%M-%S')
                    Log.e("System cannot access internet but system time is valid", extra_details={"system_time": today_string}, flow=FLOW)
                    write_to_file(REPORTS_FOLDER, "System cannot access internet but system time is valid", extra_details={"system_time": today_string})
        else:
            today = datetime.today()
            today_string = today.strftime('%y-%m-%d-%H-%M-%S')
            Log.i("System time is valid", extra_details={"system_time": today_string}, flow=FLOW)
            write_to_file(REPORTS_FOLDER, "System time is valid", extra_details={"system_time": today_string})
            save_rtc_time()
    else:
        # LEGACY CODE
        Log.i("RTC not installed, using legacy method", flow=FLOW)
        write_to_file(REPORTS_FOLDER, "RTC not installed, using legacy method")
        fix_time_without_rtc(should_fix)
