from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from me_core.memory import EpisodicMemory, JsonlMemoryStorage, SemanticMemory
from me_core.types import AgentEvent, EventKind


class MemoryPersistenceTestCase(unittest.TestCase):
    def test_episode_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = JsonlMemoryStorage(Path(tmpdir) / "episodes.jsonl")
            mem = EpisodicMemory(storage, max_episodes=10)
            ep = mem.begin_episode(1)
            event = AgentEvent.now(
                event_type=EventKind.PERCEPTION.value,
                payload={"raw": {"text": "hello"}},
                kind=EventKind.PERCEPTION,
            )
            mem.end_episode(ep, 1, [event], "summary")

            reloaded = EpisodicMemory(storage, max_episodes=10)
            self.assertTrue(reloaded.recent_episodes())
            self.assertEqual(reloaded.recent_episodes()[0].summary, "summary")

    def test_concept_memory_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = JsonlMemoryStorage(Path(tmpdir) / "episodes.jsonl")
            semantic = SemanticMemory(storage)
            semantic.upsert_concept_memory(
                concept_id="cid_1",  # type: ignore[arg-type]
                name="demo",
                description="desc",
                example="example text",
                tags={"text"},
            )
            loaded = SemanticMemory(storage)
            cm = loaded.get("cid_1")  # type: ignore[arg-type]
            self.assertIsNotNone(cm)
            assert cm is not None
            self.assertIn("demo", cm.name)
            self.assertIn("example text", cm.examples)


if __name__ == "__main__":
    unittest.main()
