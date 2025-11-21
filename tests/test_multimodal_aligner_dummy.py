from __future__ import annotations

import unittest

from me_core.alignment.aligner import MultimodalAligner
from me_core.alignment.concepts import ConceptSpace
from me_core.alignment.embeddings import DummyEmbeddingBackend
from me_core.perception import ImagePerception, TextPerception


class MultimodalAlignerDummyTestCase(unittest.TestCase):
    """多模态 Dummy 对齐的事件级联测试。"""

    def setUp(self) -> None:
        backend = DummyEmbeddingBackend(dim=8)
        space = ConceptSpace()
        self.aligner = MultimodalAligner(
            backend=backend, concept_space=space, similarity_threshold=0.8
        )

    def test_align_text_event(self) -> None:
        event = TextPerception(split_sentences=False).perceive("苹果")[0]
        concept = self.aligner.align_event(event)

        self.assertIsNotNone(event.embedding)
        self.assertIsNotNone(concept)
        self.assertEqual(len(self.aligner.concept_space.all_concepts()), 1)

    def test_align_image_event(self) -> None:
        event = ImagePerception().perceive("dummy.png")[0]
        concept = self.aligner.align_event(event)

        self.assertIsNotNone(event.embedding)
        self.assertIsNotNone(concept)


if __name__ == "__main__":
    unittest.main()
