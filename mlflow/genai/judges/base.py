from typing import Any

from pydantic import Field, PrivateAttr

from mlflow.genai.scorers.base import Scorer
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class Judge(Scorer):
    """
    Base class for LLM-based scorers that can be aligned with human feedback.

    This class provides the foundation for judges that evaluate inputs/outputs based on
    declarative instructions. Individual judge implementations should override the
    abstract __call__ method.

    Note: This is an internal API. Users should interact with judges via:
    - Built-in judges from mlflow.genai.judges.builtin
    - The make_judge() API for creating custom judges
    """

    model: str = Field(..., description="LLM model identifier")

    _examples: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    # NB: __call__ is abstract as defined in Scorer base class
    # Individual judge implementations must override this method
