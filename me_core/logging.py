from __future__ import annotations

import logging
from typing import Iterable, Optional

from me_core.types import AgentEvent


LOGGER_NAME = "me_agent"


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """创建或获取统一的 Agent logger。"""

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def summarize_events(events: Iterable[AgentEvent]) -> str:
    """生成最近事件的简短摘要。"""

    parts = []
    for e in events:
        kind = e.kind.value if hasattr(e.kind, "value") else (e.kind or e.event_type)
        src = e.source or "-"
        parts.append(f"{kind}:{src}")
    return " | ".join(parts)


def log_step(
    logger: logging.Logger,
    step: int,
    intent_kind: str,
    reply: Optional[str],
    tool_name: Optional[str] = None,
    tool_success: Optional[bool] = None,
    events: Iterable[AgentEvent] = (),
) -> None:
    """打印每一步的关键调试信息。"""

    summary = summarize_events(events)
    msg = (
        f"[step {step}] intent={intent_kind} "
        f"events=[{summary}] "
        f"tool={tool_name or '-'}"
    )
    if tool_success is not None:
        msg += f" (success={tool_success})"
    if reply:
        msg += f" reply={reply[:120]}"
    logger.info(msg)
