# TODO: The contents of this file must be merged into mlflow.environment_variables prior to
# merging the gateway branch into master. This is being done to prevent merge conflicts with
# the periodic synchronization of master to the gateway branch. Encapsulating all code changes for
# this feature into the new directory simplifies the merge process. This file's contents
# MUST be migrated prior to final gateway branch merge and this file should be deleted during that
# merge.
from mlflow.environment_variables import _EnvironmentVariable


#: (Experimental, may be changed or removed)
#: Specifies the uri of a Mlflow Gateway Server instance to be used with the Gateway Client APIs
MLFLOW_GATEWAY_URI = _EnvironmentVariable("MLFLOW_GATEWAY_URI", str, None)
