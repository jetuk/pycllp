# content of conftest.py

import pytest
def pytest_addoption(parser):
    parser.addoption("--cl", action="store_true", help="run OpenCL tests")

def pytest_runtest_setup(item):
    if 'cl' in item.keywords and not item.config.getoption("--cl"):
        pytest.skip("need --cl option to run")
