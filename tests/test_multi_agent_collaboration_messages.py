from __future__ import annotations

import unittest

from me_core.agent.messages import TaskMessage


class TaskMessageTestCase(unittest.TestCase):
    def test_task_message_basic(self) -> None:
        msg = TaskMessage(id="1", from_role="planner", to_role="coder", kind="plan", content="do it")
        self.assertEqual(msg.from_role, "planner")
        self.assertEqual(msg.kind, "plan")


if __name__ == "__main__":
    unittest.main()
