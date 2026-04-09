"""Backward-compatible re-export shim.

This module has been moved to :mod:`mlflow.gateway.budget`.
"""

from mlflow.gateway.budget import (
    calculate_existing_cost_for_windows,
    check_budget_limit,
    fire_budget_exceeded_webhooks,
    make_budget_on_complete,
    maybe_refresh_budget_policies,
)

__all__ = [
    "calculate_existing_cost_for_windows",
    "check_budget_limit",
    "fire_budget_exceeded_webhooks",
    "make_budget_on_complete",
    "maybe_refresh_budget_policies",
]
