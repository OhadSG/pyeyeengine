import threading
from pyeyeengine.utilities.metrics import Gauge, MetricLabelsBase
import logging
import time

process_cpu_gauge = Gauge('process_cpu')
process_mem_gauge = Gauge('process_mem')
system_cpu_gauge = Gauge('system_cpu')
system_mem_gauge = Gauge('system_mem')

logger = logging.getLogger(__name__)

class ResourceUsageMetricsReporter:
    """
    Reports CPU and RAM usage per process.
    """
    def __init__(self):
        self.thread = None
        self.is_running = False

    def start(self):
        try:
            import psutil
        except ImportError:
            logger.warning('psutil is not installed, not reporting resource usage metrics')
            return

        self.thread = threading.Thread(
            target=self.__thread_entry,
            name="ResourceUsageMetricsReporter",
        )
        self.thread.daemon = True
        self.is_running = True
        self.thread.start()

    def stop(self):
        self.is_running = False

    def __thread_entry(self):
        while self.is_running:
            try:
                time.sleep(1)
                self.update()
            except:
                logger.exception("Failed to update resource usage metrics")

    def update(self):
        # the per process metrics seem to not be reliable
        # current_process_keys = self.__update_processes()
        # self.__remove_old_process_entries(current_process_keys)
        self.__update_system()

    @staticmethod
    def __update_processes():
        import psutil

        process_keys = set()
        for process in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_percent']):
            process_key = ProcessKey(process.pid, process.name())
            process_keys.add(process_key)

            process_cpu_gauge.set(process.cpu_percent(), labels=process_key)
            process_mem_gauge.set(process.memory_percent(), labels=process_key)

        return process_keys

    @staticmethod
    def __remove_old_process_entries(current_process_keys):
        for old_process_key in list(process_cpu_gauge.labels_keys - current_process_keys):
            logger.debug('Removing old CPU key: {}'.format(old_process_key))
            process_cpu_gauge.remove(old_process_key)

        for old_process_key in list(process_mem_gauge.labels_keys - current_process_keys):
            logger.debug('Removing old RAM key: {}'.format(old_process_key))
            process_mem_gauge.remove(old_process_key)

    @staticmethod
    def __update_system():
        import psutil

        for cpu_index, cpu_percent in enumerate(psutil.cpu_percent(percpu=True)):
            system_cpu_gauge.set(cpu_percent, labels={'cpu': cpu_index})

        system_mem_gauge.set(psutil.virtual_memory().percent)

class ProcessKey(MetricLabelsBase):
    def __init__(self, pid, name):
        self.pid = pid
        self.name = name

    def as_items(self):
        return (('pid', self.pid), ('name', self.name))
