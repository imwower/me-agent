from __future__ import annotations

import unittest

from me_core.policy import AgentPolicy, apply_policy_patches
from me_core.teachers.types import PolicyPatch


class PolicySchemaPatchTestCase(unittest.TestCase):
    def test_apply_patch_updates_policy(self) -> None:
        policy = AgentPolicy()
        patches = [
            PolicyPatch(target="drives", path="curiosity.min_concept_count", value=5, reason="test"),
            PolicyPatch(target="dialogue", path="dialogue.style", value="curious", reason="test"),
        ]
        new_policy = apply_policy_patches(policy, patches)
        self.assertEqual(new_policy.curiosity.min_concept_count, 5)
        self.assertEqual(new_policy.dialogue.style, "curious")


if __name__ == "__main__":
    unittest.main()
