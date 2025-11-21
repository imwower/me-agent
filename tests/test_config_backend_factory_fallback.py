from __future__ import annotations

import unittest

from me_core.alignment.embeddings import DummyEmbeddingBackend, create_embedding_backend_from_config
from me_core.config import AgentConfig


class ConfigBackendFactoryFallbackTestCase(unittest.TestCase):
    def test_fallback_to_dummy(self) -> None:
        cfg = AgentConfig(use_dummy_embedding=False, embedding_backend_module="nonexistent.module")
        backend = create_embedding_backend_from_config(cfg)
        self.assertIsInstance(backend, DummyEmbeddingBackend)


if __name__ == "__main__":
    unittest.main()
