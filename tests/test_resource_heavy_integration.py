"""
Integration test to verify resource-heavy test detection is working end-to-end.
This test intentionally allocates a large amount of memory to trigger the detection.

Skip by default to avoid memory pressure in CI. Run manually with:
  pytest tests/test_resource_heavy_integration.py -v
"""

import pytest

# Module-level variable to hold memory across test phases
_large_data = None


@pytest.fixture
def allocate_memory():
    """Fixture that allocates and holds memory"""
    global _large_data
    # Allocate approximately 600 MB of memory
    _large_data = [bytearray(100 * 1024 * 1024) for _ in range(6)]  # 6 x 100 MB = 600 MB
    # Fill with data to ensure real allocation
    for arr in _large_data:
        for i in range(0, len(arr), 1024 * 1024):
            arr[i] = i % 256
    yield _large_data
    # Clean up after test
    _large_data = None


@pytest.mark.skip(
    reason="Intentionally uses large amounts of memory; run manually to test the feature"
)
def test_high_memory_allocation(allocate_memory):
    """
    This test intentionally uses >0.5 GB of memory to trigger resource-heavy test detection.

    To run this test manually and verify resource-heavy detection works:
      pytest tests/test_resource_heavy_integration.py::test_high_memory_allocation -v -s

    Expected output should include a "Resource-heavy tests" section showing:
      tests/test_resource_heavy_integration.py::test_high_memory_allocation: mem: +0.6 GB
    """
    # Use the allocated memory
    assert len(allocate_memory) == 6
    total_size = sum(len(arr) for arr in allocate_memory)
    assert total_size == 600 * 1024 * 1024
    # Verify data is allocated
    assert allocate_memory[0][0] == 0
