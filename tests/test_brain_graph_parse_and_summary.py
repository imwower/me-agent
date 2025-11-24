from __future__ import annotations

import json
import unittest

from me_core.brain import parse_brain_graph_from_json


class BrainGraphParseTestCase(unittest.TestCase):
    def test_parse_and_summary(self) -> None:
        data = {
            "regions": [
                {"id": "r1", "name": "sensory", "kind": "sensory", "size": 100},
                {"id": "r2", "name": "motor", "kind": "motor", "size": 80},
            ],
            "connections": [
                {"id": "c1", "pre_region": "r1", "post_region": "r2", "type": "excitatory", "sparsity": 0.2}
            ],
            "metrics": [{"name": "energy", "value": 0.5, "unit": "mJ"}],
        }
        graph = parse_brain_graph_from_json("repo", json.dumps(data))
        summary = graph.summary()
        self.assertIn("区域 2", summary)
        self.assertIn("连接 1", summary)
        self.assertTrue(graph.metrics)


if __name__ == "__main__":
    unittest.main()
