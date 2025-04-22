"""Class for monitoring GPU stats on HIP devices.
Inspired by GPUMonitor, but with the pynvml method
named replaced by pyrsmi method names
"""

import contextlib
import io
import logging
import sys

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor

_logger = logging.getLogger(__name__)

is_rocml_available = False
try:
    from pyrsmi import rocml

    is_rocml_available = True
except ImportError:
    # If `pyrsmi` is not installed, a warning will be logged at monitor instantiation.
    # We don't log a warning here to avoid spamming warning at every import.
    pass


class ROCMMonitor(BaseMetricsMonitor):
    """
    Class for monitoring AMD GPU stats. This is
    class has been modified and has been inspired by
    the original GPUMonitor class written by MLflow.
    This class uses the package pyrsmi which is an
    official ROCM python package which tracks and monitor
    AMD GPU's, has been tested on AMD MI250x 128GB GPUs

    For more information see:
    https://github.com/ROCm/pyrsmi

    PyPi package:
    https://pypi.org/project/pyrsmi/


    """

    def __init__(self):
        if "pyrsmi" not in sys.modules:
            # Only instantiate if `pyrsmi` is installed.
            raise ImportError(
                "`pyrsmi` is not installed, to log GPU metrics please run `pip install pyrsmi` "
                "to install it."
            )

        try:
            rocml.smi_initialize()
        except RuntimeError:
            raise RuntimeError("Failed to initialize RSMI, skip logging GPU metrics")

        super().__init__()

        # Check if GPU is virtual. If so, collect power information from physical GPU
        self.physical_idx = []
        for i in range(rocml.smi_get_device_count()):
            try:
                self.raise_error(rocml.smi_get_device_average_power, i)
                # physical GPU if no error is raised
                self.physical_idx.append(i)
            except SystemError:
                # virtual if error is raised
                # all virtual GPUs must share physical GPU with previous virtual/physical GPU
                assert i >= 1
                self.physical_idx.append(self.physical_idx[-1])

    @staticmethod
    def raise_error(func, *args, **kwargs):
        """Raise error if message containing 'error' is printed out to stdout or stderr."""
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            func(*args, **kwargs)

        out = stdout.getvalue()
        err = stderr.getvalue()

        # Check if there is an error message in either stdout or stderr
        if "error" in out.lower():
            raise SystemError(out)
        if "error" in err.lower():
            raise SystemError(err)

    def collect_metrics(self):
        # Get GPU metrics.
        self.num_gpus = rocml.smi_get_device_count()

        for i in range(self.num_gpus):
            memory_used = rocml.smi_get_device_memory_used(i)
            memory_total = rocml.smi_get_device_memory_total(i)
            self._metrics[f"gpu_{i}_memory_usage_percentage"].append(
                round(memory_used / memory_total * 100, 1)
            )
            self._metrics[f"gpu_{i}_memory_usage_gigabytes"].append(memory_used / 1e9)

            device_utilization = rocml.smi_get_device_utilization(i)
            self._metrics[f"gpu_{i}_utilization_percentage"].append(device_utilization)

            power_watts = rocml.smi_get_device_average_power(self.physical_idx[i])
            power_capacity_watts = 500  # hard coded for now, should get this from rocm-smi
            self._metrics[f"gpu_{i}_power_usage_watts"].append(power_watts)
            self._metrics[f"gpu_{i}_power_usage_percentage"].append(
                (power_watts / power_capacity_watts) * 100
            )

            # TODO:
            # memory_busy (and other useful metrics) are available in pyrsmi>1.1.0.
            # We are currently on pyrsmi==1.0.1, so these are not available
            # memory_busy  = rocml.smi_get_device_memory_busy(i)
            # self._metrics[f"gpu_{i}_memory_busy_time_percent"].append(memory_busy)

    def aggregate_metrics(self):
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}

    def __del__(self):
        if is_rocml_available:
            rocml.smi_shutdown()
