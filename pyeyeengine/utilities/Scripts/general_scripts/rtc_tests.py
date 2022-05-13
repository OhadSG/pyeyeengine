import os
import subprocess as sp
import time
from datetime import date, datetime, timezone

# Constants
RTC_I2C_ADDRESS = 0x68

def rtc_installed():
    p = sp.Popen(['i2cget', '-y', '-f', '1', '{}'.format(RTC_I2C_ADDRESS)], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    output, catch = p.communicate(b"input data that is passed to subprocess' stdin")

    if "Error: Read failed" in catch.decode("utf-8"):
        return False
    elif "resource busy" in catch.decode("utf-8"):
        return False
    else:
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
        print("Saved time to RTC: {}".format(today))
        return True
    except Exception as e:
        print("Could not save time to RTC: {}".format(e))
        return False

def get_rtc_time():
    date = []
    os.system("i2cset -y -f 1 {} 0x0".format(RTC_I2C_ADDRESS))

    for i in range(8):
        p = sp.Popen(['i2cget', '-y', '-f', '1', '{}'.format(RTC_I2C_ADDRESS)], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        output, catch = p.communicate(b"input data that is passed to subprocess' stdin")
        date.append(int(output.decode("utf-8"), 16))

    read_date_string = '{}-{}-{}-{}-{}-{}'.format(date[6], date[5], date[4], date[2], date[1], date[0] if date[0] < 60 else 0)

    try:
        return datetime.strptime(read_date_string, '%y-%m-%d-%H-%M-%S')
    except:
        return datetime.strptime("70-01-01-00-00-00", '%y-%m-%d-%H-%M-%S')

if __name__ == '__main__':
    print(rtc_installed())
    # print(save_rtc_time())
    print(get_rtc_time())