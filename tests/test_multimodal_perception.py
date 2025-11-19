from __future__ import annotations

import unittest

from me_core.perception import MultiModalPerception, ImagePerception
from me_core.types import MultiModalInput, ImageRef


class MultiModalPerceptionTestCase(unittest.TestCase):
    """多模态感知模块的输入输出结构测试。"""

    def test_text_only_input(self) -> None:
        p = MultiModalPerception()
        event = p.perceive("你好，多模态世界")

        self.assertEqual(event.event_type, "perception")
        self.assertEqual(event.modality, "text")
        self.assertIn("text", event.tags)
        payload = event.payload or {}
        self.assertEqual(payload.get("kind"), "perception")
        raw = payload.get("raw") or {}
        self.assertEqual(raw.get("text"), "你好，多模态世界")

    def test_text_plus_image_input(self) -> None:
        p = MultiModalPerception()
        mm = MultiModalInput(
            text="这张图片里有什么？",
            image_meta={"path": "examples/apple.png"},
        )
        event = p.perceive(mm)

        self.assertEqual(event.event_type, "perception")
        # 单文本 + 单图像 → mixed 模态
        self.assertEqual(event.modality, "mixed")
        self.assertIn("image", event.tags)
        payload = event.payload or {}
        self.assertIn("embeddings", payload)
        self.assertIn("modalities", payload)
        self.assertIn("image_meta", payload.get("raw") or {})


class ImagePerceptionTestCase(unittest.TestCase):
    """图像感知模块的基本行为测试。"""

    def test_image_path_input(self) -> None:
        p = ImagePerception()
        event = p.perceive("examples/dummy.png")

        self.assertEqual(event.event_type, "perception")
        self.assertEqual(event.modality, "image")
        self.assertIn("image", event.tags)


if __name__ == "__main__":
    unittest.main()

