"""生成长期自我总结报告，便于人工检查 Agent 状态。"""

from __future__ import annotations

import argparse
from pathlib import Path

from me_core.config import load_agent_config
from me_core.memory import EpisodicMemory, SemanticMemory, JsonlMemoryStorage
from me_core.self_model import SimpleSelfModel
from me_core.self_model.self_report import generate_long_term_report
from me_core.world_model import SimpleWorldModel


def main() -> None:
    parser = argparse.ArgumentParser(description="生成长期自我总结报告")
    parser.add_argument("--agent-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="reports")
    args = parser.parse_args()

    cfg = load_agent_config(args.agent_config)
    storage = JsonlMemoryStorage(
        Path(cfg.episodes_path or "outputs/episodes.jsonl"),
        Path(cfg.concepts_path) if cfg.concepts_path else None,
    )
    episodic = EpisodicMemory(storage)
    semantic = SemanticMemory(storage)
    world = SimpleWorldModel()
    self_model = SimpleSelfModel()

    report = generate_long_term_report(self_model.get_state(), world, episodic, semantic)
    print(report)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "self_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"报告已写入 {out_path}")  # noqa: T201


if __name__ == "__main__":
    main()
