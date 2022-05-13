import os.path
import pytest
from ..logging import get_serial_number


def test_get_serial_from_path(mock_serial_number_path):
    assert get_serial_number() == '211092406301814'


@pytest.fixture
def mock_serial_number_path(monkeypatch):
    monkeypatch.setenv('PYEYE_SERIAL_NUMBER_PATH', os.path.join(os.path.dirname(__file__), 'system_prop'))