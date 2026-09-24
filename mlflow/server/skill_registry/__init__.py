"""
Server-side registration and deletion flows for the Skill Registry.

These functions hold the logic that spans the registry database and MLflow artifact storage,
which are not transactional with each other. The REST layer parses requests and calls in here.
"""

from mlflow.server.skill_registry.deletion import delete_skill
from mlflow.server.skill_registry.registration import (
    SkillVersionRegistration,
    register_skill_version,
)

__all__ = ["SkillVersionRegistration", "delete_skill", "register_skill_version"]
