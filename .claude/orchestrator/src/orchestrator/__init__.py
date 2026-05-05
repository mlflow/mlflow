"""mlflow-reviewer orchestrator.

Runs the kjc9-derived adversarial checklist (alone, or paired with the spotter
discovery agent) against an MLflow PR and posts curated findings as review
comments. Triggered by a maintainer comment of `/review` on a PR.

This package is invoked from `.github/workflows/review.yml` in M1.

See README.md in this directory for architecture, cost model, and operating
notes.
"""

__version__ = "0.1.0"
