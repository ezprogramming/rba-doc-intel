"""Lightweight hook bus for instrumenting RAG + UI events."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Protocol

logger = logging.getLogger(__name__)


class HookHandler(Protocol):
    def __call__(
        self, event: str, payload: Dict[str, object]
    ) -> None:  # pragma: no cover - protocol
        """Handle a hook emission."""


class HookBus:
    """Simple pub/sub helper that keeps instrumentation decoupled from business logic."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[HookHandler]] = defaultdict(list)
        self._global: List[HookHandler] = []

    def subscribe(self, event: str, handler: HookHandler) -> None:
        self._subscribers[event].append(handler)

    def subscribe_all(self, handler: HookHandler) -> None:
        self._global.append(handler)

    def emit(self, event: str, **payload: object) -> None:
        handlers = self._subscribers.get(event, []) + self._global
        for handler in handlers:
            try:
                handler(event, payload)
            except (
                Exception
            ) as exc:  # pragma: no cover - instrumentation should never break pipeline
                logger.debug("Hook handler failed", exc_info=exc)


hooks = HookBus()


def _log_handler(event: str, payload: Dict[str, object]) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("hook %s payload=%s", event, payload)


hooks.subscribe_all(_log_handler)
