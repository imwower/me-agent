from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from me_core.alignment.embeddings import DummyEmbeddingBackend, create_embedding_backend
from me_core.config import AgentConfig, load_agent_config


class ConfigBackendTestCase(unittest.TestCase):
    def test_load_agent_config_defaults_and_override(self) -> None:
        cfg = load_agent_config(None)
        self.assertTrue(cfg.use_dummy_embedding)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cfg.json"
            path.write_text(json.dumps({"enable_curiosity": False}), encoding="utf-8")
            cfg2 = load_agent_config(str(path))
            self.assertFalse(cfg2.enable_curiosity)

    def test_create_embedding_backend_fallback(self) -> None:
        cfg = AgentConfig(use_dummy_embedding=True)
        backend = create_embedding_backend(cfg)
        self.assertIsInstance(backend, DummyEmbeddingBackend)

        cfg2 = AgentConfig(use_dummy_embedding=False)
        backend2 = create_embedding_backend(cfg2)
        self.assertIsInstance(backend2, DummyEmbeddingBackend)


if __name__ == "__main__":
    unittest.main()
