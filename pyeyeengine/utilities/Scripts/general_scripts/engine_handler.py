import glob
import sys
import json
import socket
import os
import time
import subprocess
import threading
from datetime import date, datetime, timezone

######################################## Handler ########################################

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
KICKSTART_NAME = "/server_kickstart.sh"
KICKSTART_PATH = FILE_PATH + KICKSTART_NAME
CRONJOB = '* * * * * /usr/bin/python3 -u /usr/local/lib/python3.5/dist-packages/pyeyeengine/utilities/Scripts/general_scripts/engine_handler.py'

def open_socket(timeout=10):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 5006))
    sock.settimeout(timeout)
    MESSAGE_SIZE_INDICATOR_LENGTH = 4
    return SocketWrapper(sock)

def check_files():
    print_status("Looking for pyeyeengine files...")

    pip3 = False
    remote = False

    exists = os.path.exists("/usr/local/lib/python3.5/dist-packages/pyeyeengine/")

    if exists == False:
        print_fail("Engine files not found")
    else:
        pip3_installed = os.popen('pip3 list | grep "[p]yeyeengine"').read()

        if pip3_installed is not '':
            pip3 = True
            print_success("Engine files found installed with pip3: {}".format(pip3_installed))
        else:
            remote = True
            print_success("Engine files found installed from remote machine")

    return exists, pip3, remote

def count_instances():
    remote_count = int(os.popen('ps aux | grep "/[p]yeyeengine_server" | wc -l').read())
    pip3_count = int(os.popen('ps aux | grep "/[p]yeyeengine-server" | wc -l').read())
    return remote_count + pip3_count

def check_running():
    print_status("Looking for pyeyeengine-server process...")

    is_running = False

    count_result = count_instances()

    if count_result > 1:
        print_fail("Too many engines are running, killing and restarting...")
        kill_engines()
        return False

    psaux_reponse1 = os.popen("ps aux | grep '[p]yeyeengine-server'").read()
    psaux_reponse2 = os.popen("ps aux | grep '[p]yeyeengine_server.py'").read()

    print_status("Engine instances: \033[0;32m{}".format(count_result))

    not_running_message = "\033[0;31mNot Running\033[0m"

    if psaux_reponse1 is not None and psaux_reponse1 is not '' or \
            psaux_reponse2 is not None and psaux_reponse2 is not '':
        is_running = True

    print("Installed: {}".format(
        not_running_message if psaux_reponse1 is None or psaux_reponse1 == '' else "\033[0;32m{}\033[0m".format(
            psaux_reponse1)))
    print("Remote: {}".format(
        not_running_message if psaux_reponse2 is None or psaux_reponse2 == '' else "\033[0;32m{}\033[0m".format(
            psaux_reponse2)))

    return is_running

def kill_engines():
    report_to_admin("Killing all running engines")
    os.popen('pkill -f /usr/local/lib/python3.5/dist-packages/pyeyeengine/server/pyeyeengine_server.py')
    os.popen('killall pyeyeengine-server')
    return True

def check_ping():
    print_status("Pinging pyeyeengine-server...")

    try:
        sw = open_socket()
        sw.send(json.dumps({'name': "get_monitor_data"}))
        message = json.loads(sw.receive_message().decode("utf-8"))
        print_success("Status: {}".format(message["data"]))
        return True
    except Exception as e:
        print_fail("Status: Engine is not responding\nError: {}".format(e))
        return False

def run_engine(pip3=False, remote=False):
    if count_instances() > 0:
        return False

    now = str(datetime.now())

    print_status("Starting server...")

    os.system("rm /root/engine_run_log.txt")

    if pip3:
        pip3_kickstart_script = "./server_kickstart_pip3.sh"

        if os.path.exists("{}".format(pip3_kickstart_script)) is False:
            print_status("Creating pip3 run script")
            with open("{}".format(pip3_kickstart_script), "w") as script:
                script.write('echo "-> Engine Kickstart Script (PIP3 Installed)"\n')
                script.write('nohup pyeyeengine-server >> ~/engine_run_log.txt 2>&1 &')
                script.close()

        os.system("chmod +x {} && {}".format(pip3_kickstart_script, pip3_kickstart_script))
    elif remote:
        remote_kickstart_script = "./server_kickstart_remote.sh"

        if os.path.exists("{}".format(remote_kickstart_script)) is False:
            print_status("Creating remote run script")
            with open("{}".format(remote_kickstart_script), "w") as script:
                script.write('echo "-> Engine Kickstart Script (Remote Installed)"\n')
                script.write('nohup python3 -u /usr/local/lib/python3.5/dist-packages/pyeyeengine/server/pyeyeengine_server.py >> ~/engine_run_log.txt 2>&1 &')
                script.close()

        os.system("chmod +x {} && {}".format(remote_kickstart_script, remote_kickstart_script))

    with open("/root/engine_run_check.txt", "a+") as output_log:
        output_log.write("{} Engine check: Engine was not running, running it now.\n".format(now))
        output_log.close()

def print_status(msg):
    print("\n\033[0;93m{}\033[0m\n".format(msg))

def print_success(msg):
    print("\033[0;32m{}\033[0m".format(msg))

def print_fail(msg):
    print("\033[0;31m{}\033[0m".format(msg))

def prepare_admin_report():
    now = str(datetime.now())

    with open("/etc/admin_report.txt", "w+") as report:
        report.write("{} Engine Handler Results:\n".format(now))
        report.close()

def report_to_admin(msg):
    now = str(datetime.now())

    with open("/etc/admin_report.txt", "a+") as report:
        report.write("{} {}\n".format(now, msg))
        report.close()

def check_cronjob():
    crontab = os.popen('crontab -l').read()
    print(crontab)

    if 'no crontab' in crontab or crontab.count(CRONJOB) == 0:
        os.popen('crontab -r')
        return False
    elif crontab.count(CRONJOB) == 1:
        return True
    else:
        os.popen('crontab -r')
        return False

def setup_cronjob():
    # Check how to trigger!
    # service cron status
    # service cron start
    os.system('(crontab -l 2>/dev/null; echo "{}") | crontab -'.format(CRONJOB))

def remove_cronjob():
    os.system("crontab -r")

def is_crontab_enabled():
    status = os.popen('service cron status').read()
    print(status)

    if 'is not running' in status:
        return False

    return True

def start_cronjob():
    os.system('service cron start')

def stop_cronjob():
    os.system('service cron stop')

def cronjob_handler():
    now = str(datetime.now())

    if is_crontab_enabled() == False:
        start_cronjob()
        cronjob_handler()
        return

    if check_cronjob():
        print_success("Cronjob is installed correctly")
        with open("/root/engine_run_check.txt", "a+") as output_log:
            output_log.write("{} Crontab is installed and running\n".format(now))
            output_log.close()
    else:
        print_fail("Cronjob is not installed correctly, handling...")
        setup_cronjob()
        cronjob_handler()

def engine_handler(files_exist, pip3, remote):
    now = str(datetime.now())

    if check_running():
        report_to_admin("Engine process is running")

        if check_ping():
            report_to_admin("Engine is responding")
            report_to_admin("Engine is ready")
            print_success("\n\n\n*** Engine is installed and running ***\n\n\n")

            with open("/root/engine_run_check.txt", "a+") as output_log:
                output_log.write("{} Engine check: Engine is running\n".format(now))
                output_log.close()
        else:
            report_to_admin("Engine is not responding to ping")
            print_status("Engine is not responding to ping")
            kill_engines()
            time.sleep(5)
            engine_handler(files_exist, pip3, remote)
    else:
        report_to_admin("Running engine...")
        run_engine(pip3, remote)
        print_status("Will check engine again in 10 seconds...")
        time.sleep(10)
        engine_handler(files_exist, pip3, remote)


######################################## Socket Wrapper ########################################

MESSAGE_SIZE_INDICATOR_LENGTH = 4

class SocketWrapper:
    def __init__(self, socket) -> None:
        self._socket = socket

    def send(self, message_bytes):
        try:
            message_length = len(message_bytes)
            self._socket.send(message_length.to_bytes(MESSAGE_SIZE_INDICATOR_LENGTH, byteorder='big') + message_bytes.encode())
            return True
        except:
            return False

    def receive_message(self):
        try:
            return self._try_receive_message()
        except ConnectionResetError:
            return None

    def _try_receive_message(self):
        request_length = self._read_message_length()
        return self._read_n_bytes(request_length)

    def _read_message_length(self):
        return int.from_bytes(self._read_n_bytes(MESSAGE_SIZE_INDICATOR_LENGTH), byteorder='big', signed=True)

    def _read_n_bytes(self, n):
        data = b''
        while len(data) < n:
            packet = self._socket.recv(n - len(data))
            if not packet:
                raise ConnectionResetError
            data += packet
        return data

######################################## Main ########################################

if __name__ == '__main__':
    prepare_admin_report()

    files_exist, pip3, remote = check_files()

    if files_exist:
        report_to_admin("Engine files found")
        engine_handler(files_exist, pip3, remote)
    else:
        report_to_admin("Engine is not installed")
        print_fail("\n\n\n*** An engine needs to be installed on this system ***\n\n\n")

    # cronjob_handler()
