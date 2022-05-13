import typing
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import json
import sys
import subprocess
import logging
from threading import Lock
from pyeyeengine.utilities.preferences import EnginePreferences

logger = logging.getLogger(__name__)

def init_metrics(preferences: EnginePreferences = EnginePreferences.getInstance()):
    if not preferences.metrics_enabled:
        logger.info('Metrics disabled')
        return

    port = 9000
    logger.info('Starting metrics server on port {}'.format(port))

    server = HTTPServer(('0.0.0.0', port), MetricsRequestHandler)
    threading.Thread(
        name='MetricsServer',
        target=server.serve_forever,
        daemon=True,
    ).start()

    threading.Thread(
        name='StartExportersThread',
        target=start_exporters,
        daemon=True,
    ).start()


def start_exporters():
    logger.info('Starting exporters')
    try:
        subprocess.run(
            args=['start-exporters'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=True,
        )
    except FileNotFoundError:
        logger.warning('start-exporters script not found, not running exporters')
    except:
        logger.exception('Failed to start exporters')

metrics = []

class MetricsRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        res = '\n'.join(
            str(metric)
            for metric in metrics
        )
        self.send_response(200)
        self.send_header("Content-type", "text")
        self.end_headers()
        self.wfile.write(res.encode('utf8'))
        self.wfile.write(b'\n')


class Gauge:
    def __init__(self, name, namespace=None):
        self.name = name
        self.namespace = namespace
        self.labels_to_value = {}
        self.lock = Lock()
        metrics.append(self)

    @property
    def labels_keys(self):
        return self.labels_to_value.keys()

    def set(self, value, labels=None):
        key = MetricLabelsBase.from_input(labels)
        with self.lock:
            self.labels_to_value[key] = value

    def remove(self, labels):
        key = MetricLabelsBase.from_input(labels)
        with self.lock:
            self.labels_to_value.pop(key, None)

    def clear(self):
        with self.lock:
            self.labels_to_value.clear()

    def __str__(self):
        if self.namespace is None:
            qualified_name = self.name
        else:
            qualified_name = '_'.join([self.namespace, self.name])

        with self.lock:
            return '\n'.join(
                format_metric_row(qualified_name, labels, value)
                for labels, value in self.labels_to_value.items()
            )


def format_metric_row(qualified_name: str, labels: 'MetricLabelsBase', value: float):
    labels_items = tuple(labels.as_items())
    if len(labels_items) == 0:
        return '{} {}'.format(qualified_name, value)
    else:
        labels = ','.join(
            '{}={}'.format(key, format_value(value))
            for key, value in labels_items
        )

        return '{}{{{}}} {}'.format(qualified_name, labels, value)


class MetricLabelsBase:
    """
    Base class for metric labels stored as a key in each gauge.
    """
    def as_items(self) -> typing.Iterable[typing.Tuple[str, typing.Any]]:
        raise NotImplementedError()

    def __str__(self):
        return ','.join(
            '{}={}'.format(key, format_value(value))
            for key, value in self.as_items()
        )

    def __hash__(self):
        return hash(tuple(self.as_items()))

    def __eq__(self, other):
        return isinstance(other, MetricLabelsBase) and tuple(self.as_items()) == tuple(other.as_items())

    @staticmethod
    def from_input(labels) -> 'MetricLabelsBase':
        if isinstance(labels, MetricLabelsBase):
            return labels
        elif labels is None:
            return DictMetricsLabels({})
        elif isinstance(labels, dict):
            return DictMetricsLabels(labels)
        else:
            raise Exception('Unsupported metric labels: {}'.format(labels))


class DictMetricsLabels(MetricLabelsBase):
    def __init__(self, labels: dict):
        self.items = tuple(labels.items())

    def as_items(self):
        return self.items


class Counter(Gauge):
    def __init__(self, name, namespace, reset_at=sys.maxsize):
        super().__init__(name, namespace)
        self.reset_at = reset_at

    def inc(self, labels=None):
        self.inc_by(1, labels)

    def inc_by(self, increase_by, labels=None):
        key = MetricLabelsBase.from_input(labels)
        with self.lock:
            value = self.labels_to_value.get(key)
            if value is None:
                value = increase_by
            else:
                if value > self.reset_at - increase_by:
                    value = 0
                else:
                    value += increase_by
            self.labels_to_value[key] = value


def format_value(value):
    if isinstance(value, (int, float)):
        return '"' + str(value) + '"'
    elif isinstance(value, str):
        return json.dumps(value)
    else:
        raise Exception('unknown value type: {}'.format(type(value)))
