"""
This module imports contents from CloudPickle in a way that is compatible with the
``pickle_module`` parameter of PyTorch's model persistence function: ``torch.save``
(see https://github.com/pytorch/pytorch/blob/692898fe379c9092f5e380797c32305145cd06e1/torch/
serialization.py#L192). It is included as a distinct module from :mod:`mlflow.pytorch` to avoid
polluting the namespace with wildcard imports.

Calling ``torch.save(..., pickle_module=mlflow.pytorch.pickle_module)`` will persist PyTorch model
definitions using CloudPickle, leveraging improved pickling functionality such as the ability
to capture class definitions in the "__main__" scope.

TODO: Remove this module or make it an alias of CloudPickle when CloudPickle and PyTorch have
compatible pickling APIs.
"""

# Import all contents of the CloudPickle module in an attempt to include all functions required
# by ``torch.save``.


# CloudPickle does not include `Unpickler` in its namespace, which is required by PyTorch for
# deserialization. Noting that CloudPickle's `load()` and `loads()` routines are aliases for
# `pickle.load()` and `pickle.loads()`, we therefore import Unpickler from the native
# Python pickle library.
from pickle import Unpickler  # noqa: F401

from cloudpickle import *  # noqa: F403

# PyTorch uses the ``Pickler`` class of the specified ``pickle_module``
# (https://github.com/pytorch/pytorch/blob/692898fe379c9092f5e380797c32305145cd06e1/torch/
# serialization.py#L290). Unfortunately, ``cloudpickle.Pickler`` is an alias for Python's native
# pickling class: ``pickle.Pickler``, instead of ``cloudpickle.CloudPickler``.
# https://github.com/cloudpipe/cloudpickle/pull/235 has been filed to correct the issue,
# but this import renaming is necessary until either the requested change has been incorporated
# into a CloudPickle release or the ``torch.save`` API has been updated to be compatible with
# the existing CloudPickle API.
from cloudpickle import CloudPickler as Pickler  # noqa: F401
