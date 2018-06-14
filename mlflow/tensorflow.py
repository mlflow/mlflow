"""MLflow integration for Tensorflow."""

from __future__ import absolute_import


class _TFWrapper(object):
    """
    Wrapper class that creates a predict function such that 
    predict(data: pandas.DataFrame) -> pandas.DataFrame
    """
    def __init__(self, saved_model_dir, signature_def_name=None):
        self._saved_model_dir = saved_model_dir
        self._signature_def_name = signature_def_name
