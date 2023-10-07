import threading
from collections import deque

from mlflow.utils.validation import (
    MAX_METRICS_PER_BATCH,
    MAX_PARAM_VAL_LENGTH,
    MAX_TAG_VAL_LENGTH,
)


class RunBatch:
    def __init__(self, params, tags, metrics) -> None:
        self.id = -1
        self.params = params if params else []
        self.tags = tags if tags else []
        self.metrics = metrics if metrics else []


class PendingRunBatches:
    def __init__(self, run_id) -> None:
        self.run_id = run_id
        self.pending = deque()
        self.enqueue_id = 0
        self.lock = threading.Lock()

    def _get_next_id(self):
        self.lock.acquire()
        next_id = self.enqueue_id + 1
        self.enqueue_id = next_id
        self.lock.release()
        return next_id

    def append(self, batch: RunBatch) -> int:
        batch.id = self._get_next_id()
        self.pending.append(batch)
        return batch.id

    def is_empty(self) -> bool:
        return len(self.pending) == 0

    def get_next_batch(
        self,
        max_tags=MAX_TAG_VAL_LENGTH,
        max_params=MAX_PARAM_VAL_LENGTH,
        max_metrics=MAX_METRICS_PER_BATCH,
    ):
        params_batch = []
        tags_batch = []
        metrics_batch = []
        start_watermark = -1
        end_watermark = -1
        while len(self.pending) > 0:
            next_batch = self.pending[0]  # RunBatch
            if (
                len(params_batch) + len(next_batch.params) > max_params
                or len(params_batch) + len(next_batch.tags) > max_tags
                or len(metrics_batch) + len(next_batch.metrics) > max_metrics
            ):
                break
            params_batch += next_batch.params or []
            tags_batch += next_batch.tags or []
            metrics_batch += next_batch.metrics or []
            start_watermark = next_batch.id if start_watermark == -1 else start_watermark
            end_watermark = next_batch.id
            self.pending.popleft()

        return (start_watermark, end_watermark, params_batch, tags_batch, metrics_batch)
