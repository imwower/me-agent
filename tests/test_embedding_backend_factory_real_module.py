from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

from me_core.alignment.embeddings import DummyEmbeddingBackend, create_embedding_backend_from_config
from me_core.config import AgentConfig


class EmbeddingBackendFactoryRealModuleTestCase(unittest.TestCase):
    def test_factory_loads_custom_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = Path(tmpdir) / "real_mod.py"
            module_path.write_text(
                "from me_core.alignment.embeddings import DummyEmbeddingBackend\n"
                "def create_backend(cfg):\n"
                "    return DummyEmbeddingBackend(dim=16)\n",
                encoding="utf-8",
            )
            sys.path.insert(0, tmpdir)
            cfg = AgentConfig(use_dummy_embedding=False, embedding_backend_module="real_mod", embedding_backend_kwargs={"dim": 16})
            backend = create_embedding_backend_from_config(cfg)
            self.assertIsInstance(backend, DummyEmbeddingBackend)
            sys.path.remove(tmpdir)

    def test_factory_fallback_on_error(self) -> None:
        cfg = AgentConfig(use_dummy_embedding=False, embedding_backend_module="nonexistent.module")
        backend = create_embedding_backend_from_config(cfg)
        self.assertIsInstance(backend, DummyEmbeddingBackend)


if __name__ == "__main__":
    unittest.main()
