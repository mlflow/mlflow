"""Class for monitoring process-level stats."""

import logging
import os

import psutil

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor

_logger = logging.getLogger(__name__)


class ProcessMonitor(BaseMetricsMonitor):
    """Monitor metrics for the current process and optionally its children.

    This monitor tracks CPU, memory, thread count, and open file descriptors
    for the current Python process. When `include_children` is True, metrics
    from all child processes are aggregated.
    """

    def __init__(self, include_children: bool = True):
        """Initialize the ProcessMonitor.

        Args:
            include_children: If True, aggregate metrics from child processes.
        """
        super().__init__()
        self._include_children = include_children
        try:
            self._process = psutil.Process(os.getpid())
            # Initialize cpu_percent to avoid first-call returning 0
            self._process.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            _logger.warning(f"Failed to initialize ProcessMonitor: {e}")
            self._process = None

    def _get_processes(self):
        """Get the list of processes to monitor."""
        if self._process is None:
            return []

        processes = [self._process]
        if self._include_children:
            try:
                processes.extend(self._process.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return processes

    def collect_metrics(self):
        """Collect process-level metrics."""
        if self._process is None:
            return

        processes = self._get_processes()
        if not processes:
            return

        # Aggregate metrics across all processes
        total_cpu_percent = 0.0
        total_memory_rss = 0
        total_memory_vms = 0
        total_threads = 0
        total_open_files = 0

        for proc in processes:
            try:
                # CPU percentage
                total_cpu_percent += proc.cpu_percent()

                # Memory info
                mem_info = proc.memory_info()
                total_memory_rss += mem_info.rss
                total_memory_vms += mem_info.vms

                # Thread count
                total_threads += proc.num_threads()

                # Open file descriptors (not available on all platforms)
                try:
                    total_open_files += proc.num_fds()
                except AttributeError:
                    # num_fds() not available on Windows, use num_handles() instead
                    try:
                        total_open_files += proc.num_handles()
                    except AttributeError:
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process may have terminated or we don't have access
                continue

        # Get memory percentage relative to total system memory
        try:
            memory_percent = self._process.memory_percent()
            if self._include_children:
                # Recalculate based on total RSS
                total_memory = psutil.virtual_memory().total
                memory_percent = (total_memory_rss / total_memory) * 100 if total_memory > 0 else 0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            memory_percent = 0

        # Store metrics
        self._metrics["process_cpu_percentage"].append(total_cpu_percent)
        self._metrics["process_memory_rss_megabytes"].append(total_memory_rss / 1e6)
        self._metrics["process_memory_vms_megabytes"].append(total_memory_vms / 1e6)
        self._metrics["process_memory_percentage"].append(memory_percent)
        self._metrics["process_threads"].append(total_threads)
        self._metrics["process_open_files"].append(total_open_files)

    def aggregate_metrics(self):
        """Aggregate collected metrics by averaging."""
        aggregated = {}
        for key, values in self._metrics.items():
            if values:
                aggregated[key] = round(sum(values) / len(values), 1)
        return aggregated
