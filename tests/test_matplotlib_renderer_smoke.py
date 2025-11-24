from __future__ import annotations

import os
import tempfile
import unittest

from me_core.research.plot_types import PlotSpec, LineSeries
from me_ext.plots.matplotlib_backend import PlotRenderer


class MatplotlibRendererTestCase(unittest.TestCase):
    def test_render_line(self) -> None:
        spec = PlotSpec(
            id="test_line",
            kind="line",
            title="line",
            x_label="x",
            y_label="y",
            line_series=[LineSeries(label="a", x=[0, 1], y=[0, 1])],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            renderer = PlotRenderer(tmpdir)
            path = renderer.render(spec)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
