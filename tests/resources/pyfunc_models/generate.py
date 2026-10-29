from pathlib import Path

import cloudpickle
from backward_compatibility_model import MyModel

import mlflow

model_dir = Path(__file__).parent
mlflow.pyfunc.save_model(
    python_model=MyModel(),
    code_path=[str(model_dir / "backward_compatibility_model.py")],
    path=str(model_dir / mlflow.__version__),
    pip_requirements=[
        f"mlflow=={mlflow.__version__}",
        f"cloudpickle=={cloudpickle.__version__}",
    ],
)
