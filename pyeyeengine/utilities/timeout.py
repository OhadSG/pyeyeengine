from threading import Thread, Event
from typing import Callable

class Timeout:
    def __init__(self, timeout: float, handler: Callable[[float], None]):
        self.timeout = timeout
        self.handler = handler
        self.done_event = Event()
        self.waiter_thread = Thread(target=self.__thread_main, name='TimeoutWaiterThread', daemon=True)

    def __thread_main(self):
        did_finish = self.done_event.wait(self.timeout)
        if not did_finish:
            self.handler(self.timeout)

    def __enter__(self):
        self.waiter_thread.start()

        return self

    def __exit__(self, type, value, traceback):
        self.done_event.set()