# The 2 importing commands are for testing these imported library
# code files won't be captured by `infer_code_paths=True`
import scipy
import sklearn

sk_version = sklearn.__version__
scipy_version = scipy.__version__
