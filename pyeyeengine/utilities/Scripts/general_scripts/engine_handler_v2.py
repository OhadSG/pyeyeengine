import glob
import sys
import json
import socket
import os
import time
import subprocess
import threading
from enum import Enum
from datetime import date, datetime, timezone

############# GLOBALS #############

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
KICKSTART_NAME = "/server_kickstart.sh"
KICKSTART_PATH = FILE_PATH + KICKSTART_NAME
CRONJOB = '* * * * * /usr/bin/python3 -u /usr/local/lib/python3.5/dist-packages/pyeyeengine/utilities/Scripts/general_scripts/engine_handler.py'

class ErrorCode(Enum):
    FILES_EXCEPTION = {"code": 1, "message": "EngineFilesNotFound"}
    RUN_EXCEPTION = {"code": 2, "message": "EngineRunFailed"}
    TERMINATION_EXCEPTION = {"code": 3, "message": "EngineKillFailed"}

class EngineException(Exception):
    def __init__(self, message=""):
        super().__init__(message)

############# UTILITIES #############

def open_socket(timeout=10):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 5006))
    sock.settimeout(timeout)
    MESSAGE_SIZE_INDICATOR_LENGTH = 4
    return SocketWrapper(sock)

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

############# SOCKET WRAPPER #############

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

############# IMPLEMENTATION #############

class State(object):
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.on_success_index = 0
        self.on_fail_index = 0
        self.sleep = 0
        self.success = False
        self.fail = False
        self.exception = None

    def perform(self, state_machine):
        assert False, "This should be implemented in a subclass"

    def raise_exception(self):
        raise self.exception

class StateMachine(object):
    def __init__(self):
        self.current_state = None
        self.states = {}
        self.pip3 = False
        self.remote = False
        self.prepare_states()

    def prepare_states(self):
        # 0
        check_files = StateCheckFiles("CheckFiles", 0)
        check_files.on_success_index = 1
        check_files.exception = EngineException(ErrorCode.FILES_EXCEPTION.value)
        self.states[check_files.index] = check_files

        # 1
        check_running = StateCheckRunning("CheckRunning", 1)
        check_running.on_success_index = 3
        check_running.on_fail_index = 2
        self.states[check_running.index] = check_running

        # 2
        build_run = StateBuildRun("BuildRun", 2)
        build_run.on_success_index = 1
        build_run.on_fail_index = 0
        build_run.exception = EngineException(ErrorCode.RUN_EXCEPTION.value)
        self.states[build_run.index] = build_run

        # 3
        check_ping = StateCheckPing("CheckPing", 3)
        check_ping.on_success_index = 5
        check_ping.on_fail_index = 4
        self.states[check_ping.index] = check_ping

        # 4
        kill_engines = StateKillEngines("KillEngines", 4)
        kill_engines.on_success_index = 1
        kill_engines.exception = EngineException(ErrorCode.TERMINATION_EXCEPTION.value)
        self.states[kill_engines.index] = kill_engines

        # 5
        done = StateDone("Done", 5)
        done.on_success_index = 99
        self.states[done.index] = done

        # self.states = {check_files.index: check_files,
        #                build_run.index: build_run,
        #                check_running.index: check_running,
        #                check_ping.index: check_ping,
        #                kill_engines.index: kill_engines,
        #                done.index: done}

    def start(self):
        self.current_state = self.states[0]
        self.manage()

    def manage(self):
        while self.current_state.success is False and \
            self.current_state.fail is False:
            print("\n-- StateMachine Tick --")
            self.current_state.perform(self)

        print("{} [Success: {}, Fail: {}]".format(self.current_state.name, self.current_state.success, self.current_state.fail))

        if self.current_state.success is True:
            if self.current_state.on_success_index == 99:
                return

            print("State Change: {}-->{}".format(self.current_state.name,
                                                   self.states[self.current_state.on_success_index].name))
            self.current_state = self.states[self.current_state.on_success_index]
            self.current_state.success = False
            self.current_state.fail = False
        elif self.current_state.fail is True:
            print("State Change: {}-->{}".format(self.current_state.name,
                                                   self.states[self.current_state.on_fail_index].name))
            self.current_state = self.states[self.current_state.on_fail_index]
            self.current_state.success = False
            self.current_state.fail = False

        self.manage()

    def count_instances():
        remote_count = int(os.popen('ps aux | grep "/[p]yeyeengine_server" | wc -l').read())
        pip3_count = int(os.popen('ps aux | grep "/[p]yeyeengine-server" | wc -l').read())
        return remote_count + pip3_count

class StateCheckFiles(State):
    def perform(self, state_machine):
        print_status("Looking for pyeyeengine files...")

        exists = os.path.exists("/usr/local/lib/python3.5/dist-packages/pyeyeengine/") and os.path.exists(
            "/usr/local/lib/python3.5/dist-packages/pyeyeengine/server/pyeyeengine_server.py")

        if exists == False:
            print_fail("Engine files not found")
            self.fail = True
            self.raise_exception()
        else:
            pip3_installed = os.popen('pip3 list | grep "[p]yeyeengine"').read()

            if pip3_installed is not '':
                self.success = True
                state_machine.pip3 = True
                print_success("Engine files found installed with pip3: {}".format(pip3_installed))
            else:
                self.success = True
                state_machine.remote = True
                print_success("Engine files found installed from remote machine")

class StateCheckRunning(State):
    def perform(self, state_machine):
        print_status("Looking for pyeyeengine-server process...")

        is_running = False

        count_result = StateMachine.count_instances()

        if count_result > 1:
            print_fail("Too many engines are running, killing and restarting...")
            self.on_fail_index = 4
            self.fail = True

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

        self.on_fail_index = 2
        self.success = is_running
        self.fail = not is_running

class StateBuildRun(State):
    def perform(self, state_machine):
        if StateMachine.count_instances() > 0:
            self.fail = True
            return

        now = str(datetime.now())

        print_status("Starting server...")

        os.system("rm /root/engine_run_log.txt")

        if state_machine.pip3:
            pip3_kickstart_script = "./server_kickstart_pip3.sh"

            if os.path.exists("{}".format(pip3_kickstart_script)) is False:
                print_status("Creating pip3 run script")
                with open("{}".format(pip3_kickstart_script), "w") as script:
                    script.write('echo "-> Engine Kickstart Script (PIP3 Installed)"\n')
                    script.write('nohup pyeyeengine-server >> /root/engine_run_log.txt 2>&1 &')
                    script.close()

            os.system("chmod +x {} && {}".format(pip3_kickstart_script, pip3_kickstart_script))
            self.success = True
        elif state_machine.remote:
            remote_kickstart_script = "./server_kickstart_remote.sh"

            if os.path.exists("{}".format(remote_kickstart_script)) is False:
                print_status("Creating remote run script")
                with open("{}".format(remote_kickstart_script), "w") as script:
                    script.write('echo "-> Engine Kickstart Script (Remote Installed)"\n')
                    script.write(
                        'nohup python3 -u /usr/local/lib/python3.5/dist-packages/pyeyeengine/server/pyeyeengine_server.py >> /root/engine_run_log.txt 2>&1 &')
                    script.close()

            os.system("chmod +x {} && {}".format(remote_kickstart_script, remote_kickstart_script))
            self.success = True

        with open("/root/engine_run_check.txt", "a+") as output_log:
            output_log.write("{} Engine check: Engine was not running, running it now.\n".format(now))
            output_log.close()

        time.sleep(15)

class StateCheckPing(State):
    def perform(self, state_machine):
        print_status("Pinging pyeyeengine-server...")

        try:
            sw = open_socket()
            sw.send(json.dumps({'name': "get_monitor_data"}))
            message = json.loads(sw.receive_message().decode("utf-8"))
            print_success("Status: {}".format(message["data"]))
            self.success = True
        except Exception as e:
            print_fail("Status: Engine is not responding\nError: {}".format(e))
            self.fail = True

class StateKillEngines(State):
    def perform(self, state_machine):
        report_to_admin("Killing all running engines")
        os.popen('pkill -f /usr/local/lib/python3.5/dist-packages/pyeyeengine/server/pyeyeengine_server.py')
        os.popen('killall pyeyeengine-server')
        self.success = True

class StateDone(State):
    def perform(self, state_machine):
        print("{}".format(self.name))
        self.success = True

############# SCRIPT #############

if __name__ == '__main__':
    print("\n*** Script Start ***")
    StateMachine().start()
    print("\n*** Script End ***")