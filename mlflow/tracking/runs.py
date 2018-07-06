class SubmittedRun(object):
    """
    Abstract class exposing information about a run submitted for execution. Note that the run ID
    may be None if it is unknown, e.g. if we launched a run against a tracking server that our
    local client cannot access.
    """
    def get_status(self):
        pass

    def wait(self):
        pass

    def run_id(self):
        pass


class LocalSubmittedRun(SubmittedRun):
    """Implementation of SubmittedRun corresponding to a local project run."""
    def __init__(self, run_id, ):
        super(LocalSubmittedRun, self).__init__()
        self._run_id = run_id

    @property
    def run_id(self):
        return self._run_id

    def get_status(self):
        pass

    def wait(self):


class DatabricksSubmittedRun(SubmittedRun):
    """Implementation of SubmittedRun corresponding to a project run on Databricks."""
    def __init__(self, run_id):
        super(DatabricksSubmittedRun, self).__init__()
        self._run_id = run_id

    @property
    def run_id(self):
        return self._run_id
