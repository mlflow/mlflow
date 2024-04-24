import threading
from typing import List

from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag


class RunBatch:
    def __init__(
        self,
        run_id: str,
        params: List[Param],
        tags: List[RunTag],
        metrics: List[Metric],
        completion_event: threading.Event,
    ) -> None:
        """Initializes an instance of `RunBatch`.

        Args:
            run_id: The ID of the run.
            params: A list of parameters.
            tags: A list of tags.
            metrics: A list of metrics.
            completion_event: A threading.Event object.
        """
        self.run_id = run_id
        self.params = params or []
        self.tags = tags or []
        self.metrics = metrics or []
        self.completion_event = completion_event
        self._exception = None

    @property
    def exception(self):
        """Exception raised during logging the batch."""
        return self._exception

    @exception.setter
    def exception(self, exception):
        self._exception = exception
