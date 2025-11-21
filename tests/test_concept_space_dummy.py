from __future__ import annotations

import unittest

from me_core.alignment.concepts import ConceptSpace


class ConceptSpaceDummyTestCase(unittest.TestCase):
    """ConceptSpace 的基础创建与复用测试。"""

    def test_reuse_nearby_concept(self) -> None:
        space = ConceptSpace()
        emb1 = [1.0, 0.0]
        emb2 = [1.0, 0.0]

        c1 = space.get_or_create(emb1, "same", threshold=0.8)
        c2 = space.get_or_create(emb2, "same_again", threshold=0.8)

        self.assertEqual(c1.id, c2.id)
        self.assertEqual(len(space.all_concepts()), 1)

    def test_create_new_concept_when_far(self) -> None:
        space = ConceptSpace()
        emb1 = [1.0, 0.0]
        emb2 = [-1.0, 0.0]

        c1 = space.get_or_create(emb1, "pos", threshold=0.9)
        c2 = space.get_or_create(emb2, "neg", threshold=0.9)

        self.assertNotEqual(c1.id, c2.id)
        self.assertGreaterEqual(len(space.all_concepts()), 2)


if __name__ == "__main__":
    unittest.main()
