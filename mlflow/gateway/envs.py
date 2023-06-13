# TODO: The contents of this file must be merged into mlflow.environment_variables prior to
# merging the gateway branch into master.
from mlflow.environment_variables import _EnvironmentVariable


#: (Experimental, may be changed or removed)
#: Specifies the uri of an Mlflow Gateway Server instance to be used with the Gateway Client APIs
MLFLOW_GATEWAY_URI = _EnvironmentVariable("MLFLOW_GATEWAY_URI", str, None)
