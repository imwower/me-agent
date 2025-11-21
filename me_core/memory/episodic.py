from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Set

from me_core.types import AgentEvent

if False:  # pragma: no cover - 类型提示占位
    from .storage import MemoryStorage


def _event_text(event: AgentEvent) -> str:
    payload = event.payload or {}
    if isinstance(payload, dict):
        raw = payload.get("raw")
        if isinstance(raw, dict) and isinstance(raw.get("text"), str):
            return raw["text"]
        if isinstance(payload.get("text"), str):
            return payload["text"]
    return ""


@dataclass
class Episode:
    id: str
    start_step: int
    end_step: int
    events: List[AgentEvent]
    summary: str
    tags: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "events": [e.to_dict() for e in self.events],
            "summary": self.summary,
            "tags": sorted(self.tags),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        events_raw = data.get("events") or []
        events: List[AgentEvent] = []
        for item in events_raw:
            if isinstance(item, dict):
                events.append(AgentEvent.from_dict(item))
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            start_step=int(data.get("start_step", 0) or 0),
            end_step=int(data.get("end_step", 0) or 0),
            events=events,
            summary=str(data.get("summary") or ""),
            tags=set(data.get("tags") or []),
            created_at=float(data.get("created_at") or time.time()),
        )


class EpisodicMemory:
    """维护可持久化的情节记忆。"""

    def __init__(self, storage: "MemoryStorage", max_episodes: int = 500) -> None:
        self.storage = storage
        self.max_episodes = max_episodes
        self._episodes: List[Episode] = []
        try:
            self._episodes = self.storage.load_episodes(limit=max_episodes)
        except Exception:
            self._episodes = []

    def begin_episode(self, start_step: int, tags: Optional[Set[str]] = None) -> Episode:
        ep = Episode(
            id=str(uuid.uuid4()),
            start_step=start_step,
            end_step=start_step,
            events=[],
            summary="",
            tags=tags or set(),
        )
        return ep

    def end_episode(
        self,
        episode: Episode,
        end_step: int,
        events: List[AgentEvent],
        summary: str,
    ) -> None:
        episode.end_step = end_step
        episode.events = list(events)
        episode.summary = summary
        if not episode.tags:
            episode.tags = set()
        self._episodes.append(episode)
        if len(self._episodes) > self.max_episodes:
            overflow = len(self._episodes) - self.max_episodes
            del self._episodes[0:overflow]
        try:
            self.storage.save_episode(episode)
        except Exception:
            pass

    def recent_episodes(self, max_count: int = 20) -> List[Episode]:
        if max_count <= 0:
            return []
        return list(self._episodes[-max_count:])

    def search(self, keyword: str, max_count: int = 50) -> List[Episode]:
        keyword_lower = keyword.lower()
        results: List[Episode] = []
        for ep in reversed(self._episodes):
            if keyword_lower in ep.summary.lower():
                results.append(ep)
            else:
                for ev in ep.events:
                    if keyword_lower in _event_text(ev).lower():
                        results.append(ep)
                        break
            if len(results) >= max_count:
                break
        return results
