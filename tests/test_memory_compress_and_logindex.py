from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from me_core.memory.episodic import EpisodicMemory, Episode
from me_core.memory.storage import JsonlMemoryStorage
from me_core.memory.log_index import LogIndex
from me_core.types import AgentEvent


class MemoryCompressLogIndexTestCase(unittest.TestCase):
    def test_compress_old_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            st = JsonlMemoryStorage(Path(tmpdir) / "eps.jsonl")
            mem = EpisodicMemory(st, max_episodes=5)
            for i in range(7):
                ep = Episode(id=str(i), start_step=i, end_step=i, events=[], summary=f"s{i}")
                mem._episodes.append(ep)
            mem.compress_old_episodes(max_keep=3)
            self.assertLessEqual(len(mem._episodes), 3)

    def test_log_index_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.jsonl"
            path.write_text(json.dumps({"kind": "experiment", "ts": 1, "scenario_id": "s1"}) + "\n", encoding="utf-8")
            idx = LogIndex(tmpdir)
            res = idx.query(kinds=["experiment"])
            self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
