import transformers
from packaging.version import Version

transformers_version = Version(transformers.__version__)
IS_NEW_FEATURE_EXTRACTION_API = transformers_version >= Version("4.27.0")
IS_TRANSFORMERS_V5_OR_LATER = transformers_version.major >= 5
