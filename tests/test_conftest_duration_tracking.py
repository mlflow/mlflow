"""
Test that duration tracking works for both flaky and non-flaky tests.
"""

import sys
import time

import pytest


@pytest.mark.flaky(attempts=2)
def test_flaky_duration_tracking():
    time.sleep(0.01)
    assert True


def test_non_flaky_duration_tracking():
    time.sleep(0.01)
    assert True


@pytest.mark.flaky(attempts=2, condition=False)
def test_flaky_with_false_condition_duration_tracking():
    time.sleep(0.01)
    assert True


@pytest.mark.flaky(attempts=2, condition=sys.platform == "nonexistent")
def test_flaky_with_false_platform_condition_duration_tracking():
    time.sleep(0.01)
    assert True
