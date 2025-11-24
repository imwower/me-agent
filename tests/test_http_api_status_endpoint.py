from __future__ import annotations

import json
import urllib.request
import unittest

from me_core.self_model import SimpleSelfModel
from me_core.world_model import SimpleWorldModel
from me_ext.http_api.server import serve_http


class HttpApiStatusTestCase(unittest.TestCase):
    def test_status_endpoint(self) -> None:
        world = SimpleWorldModel()
        self_model = SimpleSelfModel()
        thread = serve_http(world, self_model, port=8899)
        resp = urllib.request.urlopen("http://localhost:8899/status", timeout=2)
        data = json.loads(resp.read().decode("utf-8"))
        self.assertIn("self", data)
        thread.join(0.1)


if __name__ == "__main__":
    unittest.main()
