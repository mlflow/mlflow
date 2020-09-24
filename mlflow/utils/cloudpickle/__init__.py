from __future__ import absolute_import


from mlflow.utils.cloudpickle.cloudpickle import *  # noqa
from mlflow.utils.cloudpickle.cloudpickle_fast import CloudPickler, dumps, dump  # noqa

# Import an Unpickler module for compatibility with frameworks whose model serialization
# routines expect an Unpickler object to be defined (i.e. PyTorch)
from mlflow.utils.cloudpickle.mlflow_compat import Unpickler

# Conform to the convention used by python serialization libraries, which
# expose their Pickler subclass at top-level under the  "Pickler" name.
Pickler = CloudPickler

__version__ = "1.6.0"
