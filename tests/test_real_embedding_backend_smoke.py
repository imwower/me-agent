from __future__ import annotations

import os
import tempfile
from pathlib import Path
import unittest

from me_ext.backends.real_backend import RealEmbeddingBackend
from me_core.types import ImageRef


class RealEmbeddingBackendSmokeTestCase(unittest.TestCase):
    def test_stub_mode_embeddings(self) -> None:
        backend = RealEmbeddingBackend({"use_stub": True, "dim": 16})
        vecs = backend.embed_text(["苹果", "香蕉"])
        self.assertEqual(len(vecs), 2)
        self.assertAlmostEqual(sum(x * x for x in vecs[0]), 1.0, places=3)

    def test_image_hash_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "dummy.png"
            img_path.write_bytes(b"notreal")
            backend = RealEmbeddingBackend({"use_stub": True, "dim": 8})
            vecs = backend.embed_image([ImageRef(path=str(img_path))])
            self.assertEqual(len(vecs), 1)
            self.assertAlmostEqual(sum(x * x for x in vecs[0]), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
