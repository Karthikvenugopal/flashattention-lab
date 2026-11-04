import json

import pytest
import torch

from attention_lab.devices import environment, native_metal_probe


@pytest.fixture(scope="session", autouse=True)
def record_execution_environment(record_testsuite_property):
    # Keep hardware visibility tied to the process that actually ran the tests.
    record_testsuite_property("environment", json.dumps(environment()))
    record_testsuite_property("native_metal", json.dumps(native_metal_probe()))


@pytest.fixture(scope="session", autouse=True)
def single_cpu_thread():
    # Tiny matrix tests otherwise mostly measure thread-pool overhead.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pytest_addoption(parser):
    parser.addoption('--cpu-only', action='store_true', help='Explicitly skip GPU execution and GPU diagnostic probes')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--cpu-only'):
        return
    for item in items:
        device = getattr(item, 'callspec', None)
        device = device.params.get('device') if device is not None else None
        if device in ('mps', 'cuda') or item.name.startswith('test_fused_reference[') or 'doctor' in item.name:
            item.add_marker(pytest.mark.skip(reason='--cpu-only: GPU execution/probes excluded; not fresh GPU validation'))
