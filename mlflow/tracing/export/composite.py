import logging
from typing import List, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

_logger = logging.getLogger(__name__)


class CompositeSpanExporter(SpanExporter):
    """
    A composite exporter that orchestrates multiple span exporters.
    
    This exporter supports two modes:
    1. Parallel mode (default): All exporters run independently. If one fails, others continue.
    2. Sequential mode: Exporters run in order. Each must succeed before the next runs.
    
    Each exporter maintains its single responsibility while the composite handles the orchestration.
    """

    def __init__(self, exporters: List[SpanExporter], sequential: bool = False):
        """
        Initialize the CompositeSpanExporter.
        
        Args:
            exporters: List of SpanExporter instances to orchestrate.
                      Each exporter will receive the same spans.
            sequential: If True, exporters run sequentially and each must succeed
                       before the next runs. If False (default), all exporters
                       run independently.
        """
        if not exporters:
            raise ValueError("At least one exporter must be provided")
        
        self._exporters = exporters
        self._sequential = sequential
        mode = "sequential" if sequential else "parallel"
        _logger.debug(f"Initialized CompositeSpanExporter with {len(exporters)} exporters in {mode} mode")

    def export(self, spans: Sequence[ReadableSpan]):
        """
        Export spans to configured exporters.
        
        In parallel mode: All exporters receive spans independently.
        In sequential mode: Each exporter must succeed before the next runs.
        
        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects to export.
        """
        if self._sequential:
            self._export_sequential(spans)
        else:
            self._export_parallel(spans)

    def _export_parallel(self, spans: Sequence[ReadableSpan]):
        """Export spans to all exporters in parallel (original behavior)."""
        for i, exporter in enumerate(self._exporters):
            try:
                exporter.export(spans)
            except Exception as e:
                _logger.warning(
                    f"Exporter {i} ({type(exporter).__name__}) failed to export spans: {e}",
                    exc_info=True
                )

    def _export_sequential(self, spans: Sequence[ReadableSpan]):
        """Export spans to exporters sequentially, stopping on first failure."""
        for i, exporter in enumerate(self._exporters):
            try:
                exporter.export(spans)
                _logger.debug(f"Exporter {i} ({type(exporter).__name__}) succeeded")
            except Exception as e:
                _logger.warning(
                    f"Exporter {i} ({type(exporter).__name__}) failed to export spans: {e}. "
                    f"Stopping sequential export chain.",
                    exc_info=True
                )
                # In sequential mode, stop on first failure
                break

    def shutdown(self):
        """
        Shutdown all exporters.
        """
        for i, exporter in enumerate(self._exporters):
            try:
                if hasattr(exporter, 'shutdown'):
                    exporter.shutdown()
            except Exception as e:
                _logger.warning(
                    f"Failed to shutdown exporter {i} ({type(exporter).__name__}): {e}",
                    exc_info=True
                )

    def force_flush(self, timeout_millis: int = 30000):
        """
        Force flush all exporters.
        
        Args:
            timeout_millis: The maximum time to wait for the flush to complete.
        """
        for i, exporter in enumerate(self._exporters):
            try:
                if hasattr(exporter, 'force_flush'):
                    exporter.force_flush(timeout_millis)
            except Exception as e:
                _logger.warning(
                    f"Failed to flush exporter {i} ({type(exporter).__name__}): {e}",
                    exc_info=True
                )