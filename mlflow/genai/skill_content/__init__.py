"""
Shared client-side content handling for the skill registry (RFC-0008).

This package owns the pieces of the registry that touch skill *content* rather than registry
metadata: fetching Git, OCI, ZIP, and MLflow artifact sources, safe archive validation and
extraction, ``SKILL.md`` inspection, and the canonical content digest. Registration, import,
introspection, and pull flows all call into it so that content access and identity are
implemented once.

Nothing here is part of the public ``mlflow.genai`` API.
"""
