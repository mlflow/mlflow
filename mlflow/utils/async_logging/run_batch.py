import threading
from typing import List, Optional

from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag


class RunBatch:
    def __init__(
        self,
        run_id: Optional[str] = None,
        params: Optional[List["Param"]] = None,
        tags: Optional[List["RunTag"]] = None,
        metrics: Optional[List["Metric"]] = None,
        completion_event: Optional[threading.Event] = None,
    ):
        """Initializes an instance of `RunBatch`.

        Args:
            run_id (Optional[str]): The ID of the run. Default is None.
            params (Optional[List[Param]]): A list of parameters. Default is None.
            tags (Optional[List[RunTag]]): A list of tags. Default is None.
            metrics (Optional[List[Metric]]): A list of metrics. Default is None.
            completion_event (Optional[threading.Event]): A threading.Event object. Default is None.
        """
        self.run_id = run_id
        self.params = params or []
        self.tags = tags or []
        self.metrics = metrics or []
        self.completion_event = completion_event
        self._exception = None
        self.child_batches = []

    @property
    def exception(self):
        """Exception raised during logging the batch."""
        return self._exception

    @exception.setter
    def exception(self, exception):
        self._exception = exception

    def is_full(self):
        return len(self.tags) >= 100 or len(self.params) >= 100 or len(self.metrics) >= 1000

    def add_child_batch(self, child_batch):
        self.child_batches.append(child_batch)

    def __str__(self):
        return f"RunBatch(run_id={self.run_id}, metrics={self.metrics})"
