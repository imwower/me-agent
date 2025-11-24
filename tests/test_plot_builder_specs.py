from __future__ import annotations

import unittest

from me_core.brain.graph import BrainGraph
from me_core.brain.types import BrainRegion, BrainConnection
from me_core.population.types import AgentFitness
from me_core.research.plot_builder import PlotBuilder


class PlotBuilderTestCase(unittest.TestCase):
    def test_build_brain_graph_plot(self) -> None:
        g = BrainGraph(repo_id="r")
        g.add_region(BrainRegion(id="a", name="A", kind="core", size=1, meta={}))
        g.add_region(BrainRegion(id="b", name="B", kind="core", size=1, meta={}))
        g.add_connection(BrainConnection(id="c1", pre_region="a", post_region="b", type="p", sparsity=0.5, weight_scale=None, meta={}))
        spec = PlotBuilder.build_brain_graph_plot(g)
        self.assertEqual(spec.kind, "brain_graph")
        self.assertTrue(spec.graph_nodes)

    def test_build_coevo_fitness_plot(self) -> None:
        fitness = [AgentFitness(spec_id="s", scenario_scores={}, overall_score=0.5)]
        spec = PlotBuilder.build_coevo_fitness_plot(fitness)
        self.assertEqual(spec.kind, "line")
        self.assertTrue(spec.line_series)


if __name__ == "__main__":
    unittest.main()
