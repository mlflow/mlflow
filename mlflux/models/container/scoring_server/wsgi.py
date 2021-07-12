from mlflux.pyfunc import scoring_server
from mlflux import pyfunc

app = scoring_server.init(pyfunc.load_pyfunc("/opt/ml/model/"))
