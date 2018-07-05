class SubmittedRun(object):
    """
    Class exposing information about a run submitted for execution. Note that the run ID may be None
    if it is unknown, e.g. if we launched a run against a tracking server that our local client
    cannot access.
    """
    def __init__(self, run_id):
        self._run_id = run_id

    @property
    def run_id(self):
        return self._run_id
