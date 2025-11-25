import tempfile
import unittest

import torch

from me_ext.backends.real_backend import RealEmbeddingBackend


class RealBackendLoadProjectionTest(unittest.TestCase):
    def test_load_projection_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/proj.pt"
            dummy_state = {
                "text_proj": torch.nn.Linear(512, 512, bias=False).state_dict(),
                "vision_proj": torch.nn.Linear(512, 512, bias=False).state_dict(),
                "proj_dim": 512,
            }
            torch.save(dummy_state, path)
            backend = RealEmbeddingBackend({"use_stub": True, "weights_path": path, "dim": 512})
            vecs = backend.embed_text(["hello"])
            self.assertEqual(len(vecs[0]), backend.dim)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
