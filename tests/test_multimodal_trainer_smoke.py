import json
import tempfile
import unittest
from pathlib import Path

from torch.utils.data import DataLoader

from me_ext.multimodal_backend.datasets import MultimodalDataset, collate_batch
from me_ext.multimodal_backend.models import MultimodalBackbone
from me_ext.multimodal_backend.trainer import MultimodalTrainer


class MultimodalTrainerSmokeTest(unittest.TestCase):
    def test_train_small_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mm.jsonl"
            path.write_text(
                json.dumps({"id": "a", "image_path": "tests/data/dummy.png", "text": "图片", "labels": ["图"]}) + "\n",
                encoding="utf-8",
            )
            ds = MultimodalDataset([{"id": "a", "image_path": "tests/data/dummy.png", "text": "图片"}])
            loader = DataLoader(ds, batch_size=1, collate_fn=collate_batch)
            model = MultimodalBackbone()
            trainer = MultimodalTrainer(model, loader, loader, device="cpu")
            stats = trainer.train_clip_style(max_steps=5, batch_size=1, log_every=10, eval_every=10)
            self.assertGreater(len(stats.loss_history), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
