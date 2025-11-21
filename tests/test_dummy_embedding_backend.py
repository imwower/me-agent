from __future__ import annotations

import math
import unittest

from me_core.alignment.embeddings import DummyEmbeddingBackend


class DummyEmbeddingBackendTestCase(unittest.TestCase):
    """DummyEmbeddingBackend 的基础行为测试。"""

    def test_deterministic_and_normalized(self) -> None:
        backend = DummyEmbeddingBackend(dim=16)
        v1 = backend.embed_text(["hello world"])[0]
        v2 = backend.embed_text(["hello world"])[0]
        v3 = backend.embed_text(["another text"])[0]

        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, v3)

        norm = math.sqrt(sum(x * x for x in v1))
        self.assertAlmostEqual(norm, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
