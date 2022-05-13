import threading
import logging
from contextlib import contextmanager
import signal
import time
from pyeyeengine.utilities.logging import Log


logger = logging.getLogger(__name__)

# @contextmanager
def timeout(time):
    t = threading.Timer(time, raise_timeout)

    try:
        t.start()
        #yield
    except TimeoutError:
        print("did timeout")
        pass
    finally:
        t.join()

def raise_timeout():
    raise TimeoutError

class ExecTimeChecker(threading.Thread):
    def __init__(self, name="ExecTimeChecker", caller="Unknown", message=None, callback=None, should_crash=False, time_to_wait=1):
        self._stop_event = threading.Event()
        self._sleep_period = 1.0
        self._name = name
        self._caller = caller
        self._message = message
        self._counter = 0
        self._error_callback = callback
        self._should_crash = should_crash
        self._time_to_wait = time_to_wait
        threading.Thread.__init__(self, name=name)

    def run(self):
        self._counter = 0

        while True:
            if self.stopped():
                return

            self._counter = self._counter + 1

            if self._counter > self._time_to_wait:
                self.call_error_callback()
                self._stop_event.set()
                return
            else:
                self._stop_event.wait(self._sleep_period)

    def join(self, timeout=None):
        self._stop_event.set()
        threading.Thread.join(self, timeout)

    def stop(self):
        if self.stopped():
            return

        Log.d("Stopping ExecTimeChecker: {}".format(self.name))

        self._stop_event.set()

        # if threading.current_thread() is not self:
        #     threading.Thread.join(self, None)

    def stopped(self):
        return self._stop_event.is_set()

    def tick(self):
        self._counter = 0

    def call_error_callback(self):
        if self._message is not None:
            logger.info("WatchDog Triggered " + "{}".format(self._message))
            # Log.w("WatchDog Triggered", extra_details={"message": "{}".format(self._message)})

        if self._error_callback is not None:
            logger.info("WatchDog Callback Execution")
            # Log.w("WatchDog Callback Execution")
            self._error_callback()
        else:
            ExecTimeChecker.report_hanging_function(self._name, self._caller)

            if self._should_crash is True:
                raise SanityException("Function is taking too long to perform: {}".format(self._caller))

    def report_hanging_function(thread, caller):
        Log.e("Function is taking too long to perform", extra_details={"thread": thread, "caller": caller})

class SanityException(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)

class Repeater(threading.Thread):
    def __init__(self, interval, function):
        threading.Thread.__init__(self)
        self.__stopped = threading.Event()
        self.interval = interval
        self.function = function

    def run(self):
        while not self.__stopped.wait(self.interval):
            self.function()

    def stop(self):
        self.__stopped.set()