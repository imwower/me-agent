import tempfile
from pathlib import Path
import unittest

from scripts.run_small_full_loop import run_small_full_loop


class SmallFullLoopDryRunTest(unittest.TestCase):
    def test_run_small_full_loop(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        sibling_root = repo_root.parent
        snn_root = sibling_root / "self-snn"
        snn_config = snn_root / "configs" / "s0_minimal.yaml"
        snn_train_script = snn_root / "scripts" / "train_from_schedule.py"
        self.assertTrue(snn_config.exists(), "缺少 self-snn 配置文件")

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_small_full_loop(
                workspace=None,
                agent_config_path=None,
                snn_config=str(snn_config),
                snn_output=Path(tmpdir) / "snn_out",
                snn_train_script=snn_train_script,
                use_brain=False,
                use_llm_dialogue=False,
            )
            self.assertIn("benchmark_after", summary)
            self.assertIn("train_schedule", summary)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
