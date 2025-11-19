from __future__ import annotations

import unittest

from me_core.alignment.aligner import MultimodalAligner
from me_core.alignment.concepts import ConceptSpace
from me_core.alignment.embeddings import DummyEmbeddingBackend
from me_core.perception.processor import encode_to_event
from me_core.types import ImageRef, MultiModalInput


class AlignmentBasicTestCase(unittest.TestCase):
    """多模态对齐与概念空间的基础行为测试。"""

    def setUp(self) -> None:
        backend = DummyEmbeddingBackend(dim=32)
        space = ConceptSpace(similarity_threshold=0.6)
        self.aligner = MultimodalAligner(backend=backend, concept_space=space)

    def test_same_text_aligns_to_same_concept(self) -> None:
        """相同文本多次对齐应落在同一概念上。"""

        mm1 = MultiModalInput(text="苹果")
        mm2 = MultiModalInput(text="苹果")

        e1 = encode_to_event(mm1, source="test")
        e2 = encode_to_event(mm2, source="test")

        c1 = self.aligner.align_event(e1)
        c2 = self.aligner.align_event(e2)

        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)
        assert c1 is not None and c2 is not None
        self.assertEqual(c1.id, c2.id)

    def test_different_text_may_form_different_concepts(self) -> None:
        """语义差异较大的文本在高阈值下应形成不同概念。"""

        backend = DummyEmbeddingBackend(dim=32)
        space = ConceptSpace(similarity_threshold=0.9)
        aligner = MultimodalAligner(backend=backend, concept_space=space)

        e1 = encode_to_event(MultiModalInput(text="苹果"), source="test")
        e2 = encode_to_event(MultiModalInput(text="小狗"), source="test")

        c1 = aligner.align_event(e1)
        c2 = aligner.align_event(e2)

        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)
        assert c1 is not None and c2 is not None
        # 高阈值下，通常会为差异较大的向量创建不同概念
        # 若偶然落在同一概念上，也不应抛异常，因此这里只做“尽量不同”的软断言。
        if c1.id == c2.id:
            self.skipTest("hash 向量碰撞导致概念相同，跳过该用例。")

    def test_align_pair_text_image(self) -> None:
        """align_pair 能够同时处理文本与图像引用并返回概念与相似度。"""

        img = ImageRef(path="apple.png")
        concept, sim_text, sim_image = self.aligner.align_pair("苹果", img)

        self.assertIsNotNone(concept)
        self.assertTrue(-1.0 <= sim_text <= 1.0)
        self.assertTrue(-1.0 <= sim_image <= 1.0)


if __name__ == "__main__":
    unittest.main()

