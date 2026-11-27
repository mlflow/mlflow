import importlib.metadata

from packaging.version import Version

DSPY_VERSION = Version(importlib.metadata.version("dspy"))

DSPY_3_5_0_OR_NEWER = DSPY_VERSION >= Version("3.5.0")

# DSPy 3.5 asks the LM for `ChainOfThought` reasoning under a `plan` field (when the LM has no
# native reasoning) and still exposes it as `reasoning` on the prediction.
REASONING_LM_FIELD = "plan" if DSPY_3_5_0_OR_NEWER else "reasoning"
