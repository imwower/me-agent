import tempfile
import json
import unittest
from pathlib import Path

from me_ext.multimodal_backend.datasets import load_internal_multimodal, build_train_eval_splits


class MultimodalDatasetsLoaderTest(unittest.TestCase):
    def test_load_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mm.jsonl"
            path.write_text(
                json.dumps({"id": "a", "image_path": "tests/data/dummy.png", "text": "图片", "labels": ["图"]}) + "\n",
                encoding="utf-8",
            )
            data = load_internal_multimodal(str(path))
            self.assertEqual(len(data), 1)
            train, evals = build_train_eval_splits([str(path)], [])
            self.assertGreaterEqual(len(train), 0)
            self.assertIsInstance(train, list)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
