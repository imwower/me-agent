from __future__ import annotations

import unittest

from me_core.alignment.aligner import MultimodalAligner
from me_core.alignment.concepts import ConceptSpace
from me_core.alignment.embeddings import DummyEmbeddingBackend
from me_core.perception import ImagePerception, TextPerception
from me_core.self_model import SimpleSelfModel
from me_core.world_model import SimpleWorldModel


class SelfWorldIntegrationDummyTestCase(unittest.TestCase):
    """dummy 对齐驱动 world/self 统计的集成测试。"""

    def test_modalities_and_concept_stats(self) -> None:
        world = SimpleWorldModel()
        self_model = SimpleSelfModel()
        concept_space = ConceptSpace()
        aligner = MultimodalAligner(
            backend=DummyEmbeddingBackend(dim=8),
            concept_space=concept_space,
            similarity_threshold=0.8,
        )
        world.concept_space = concept_space

        world.advance_step()
        text_event = TextPerception(split_sentences=False).perceive("测试概念")[0]
        concept_text = aligner.align_event(text_event)
        world.append_event(text_event)
        self_model.observe_event(text_event, step=world.current_step)

        image_event = ImagePerception().perceive("dummy.png")[0]
        concept_image = aligner.align_event(image_event)
        world.advance_step()
        world.append_event(image_event)
        self_model.observe_event(image_event, step=world.current_step)

        self.assertIn("text", self_model.get_state().seen_modalities)
        self.assertIn("image", self_model.get_state().seen_modalities)
        self.assertTrue(world.concept_stats)

        total = sum(stats.count for stats in world.concept_stats.values())
        self.assertGreaterEqual(total, 2)

        modalities_union: set[str] = set()
        for stats in world.concept_stats.values():
            modalities_union.update(stats.modalities.keys())

        self.assertIn("text", modalities_union)
        self.assertIn("image", modalities_union)


if __name__ == "__main__":
    unittest.main()
