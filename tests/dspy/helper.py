import importlib.metadata

from packaging.version import Version

# DSPy 3.5 asks the LM for `ChainOfThought` reasoning under a `plan` field (when the LM has no
# native reasoning) and still exposes it as `reasoning` on the prediction.
REASONING_LM_FIELD = (
    "plan" if Version(importlib.metadata.version("dspy")) >= Version("3.5.0") else "reasoning"
)
