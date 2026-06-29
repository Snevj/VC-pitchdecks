"""langsmith_helper.py
Centralized LangSmith integration helpers.

Provides a safe `get_traceable()` function that returns the real
`traceable` decorator when LangSmith is enabled and installed, or a
no-op decorator when disabled/missing. This lets the project call
`@traceable(...)` everywhere without conditional imports.
"""
from __future__ import annotations

import os
from typing import Callable, Any


def _noop_traceable(name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that does nothing (no-op).

    Keeps the function signature compatible with LangSmith's `traceable`.
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return deco


def get_traceable() -> Callable[[str | None], Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """Return a `traceable` decorator.

    Behavior:
    - If `LANGSMITH_ENABLED` env var is true-ish and `langsmith` is importable,
      returns the real `langsmith.traceable` decorator.
    - Otherwise returns a no-op decorator that leaves functions unchanged.

    This allows files to safely call `@traceable(name=...)` without
    having langsmith as a hard dependency.
    """

    enabled = os.getenv("LANGSMITH_ENABLED", "false").lower() in ("1", "true")
    if not enabled:
        return _noop_traceable

    try:
        # Import lazily to avoid raising at module import time when not used.
        from langsmith import traceable as _real_traceable  # type: ignore

        return _real_traceable
    except Exception:
        return _noop_traceable


__all__ = ["get_traceable"]


# Simple LangSmith client accessor (cached)
_client = None


def get_client():
    """Return a langsmith client instance or None.

    - Returns a cached client if already created.
    - If `LANGSMITH_ENABLED` is not truthy or langsmith isn't installed,
      returns None.
    """
    global _client
    if _client is not None:
        return _client

    enabled = os.getenv("LANGSMITH_ENABLED", "false").lower() in ("1", "true")
    if not enabled:
        return None

    try:
        # langsmith exposes a `client` module with a `Client` class.
        from langsmith import client as ls_client  # type: ignore

        Client = getattr(ls_client, "Client", None)
        if Client is None:
            # If API differs, return module object for manual use
            _client = ls_client
        else:
            _client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        return _client
    except Exception:
        return None


__all__.append("get_client")

