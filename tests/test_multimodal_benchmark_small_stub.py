from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from me_core.tasks.bench_multimodal import load_multimodal_benchmark


class MultimodalBenchmarkTestCase(unittest.TestCase):
    def test_load_multimodal_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"id": "1", "image_path": "tests/data/dummy.png", "question": "?", "keywords": ["图片"]}
            path = Path(tmpdir) / "bench.jsonl"
            path.write_text(json.dumps(data), encoding="utf-8")
            scenarios = load_multimodal_benchmark(str(path))
            self.assertEqual(len(scenarios), 1)
            self.assertEqual(scenarios[0].steps[0].image_path, data["image_path"])


if __name__ == "__main__":
    unittest.main()
