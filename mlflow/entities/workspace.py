"""Workspace entity shared between server and stores."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Workspace:
    """Minimal metadata describing a workspace."""

    name: str
    description: str | None = None
