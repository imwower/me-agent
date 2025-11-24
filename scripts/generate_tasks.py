"""根据历史表现生成新任务样本并落盘。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from me_core.tasks.generated.types import TaskTemplate
from me_core.tasks.generated.generator import TaskGenerator
from me_core.memory.log_index import LogIndex
from me_core.brain.adapter import parse_brain_graph_from_json


def main() -> None:
    parser = argparse.ArgumentParser(description="生成任务样本")
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--brain-graph", type=str, default=None, help="BrainGraph JSON 字符串路径")
    parser.add_argument("--output", type=str, default="data/generated_tasks")
    parser.add_argument("--max", type=int, default=5)
    args = parser.parse_args()

    templates = [
        TaskTemplate(id="tpl_mm", kind="multimodal", description="图片问答", input_schema={}, output_schema={}, difficulty=1),
        TaskTemplate(id="tpl_code", kind="codefix", description="修复简单 bug", input_schema={}, output_schema={}, difficulty=2),
        TaskTemplate(id="tpl_brain", kind="brain_memory", description="延迟记忆任务", input_schema={}, output_schema={}, difficulty=2),
    ]
    generator = TaskGenerator(templates)
    idx = LogIndex(args.log_root)
    benchmark_results = idx.query(kinds=["experiment"], max_results=5)
    introspections = idx.query(kinds=["devloop"], max_results=5)
    brain_graph = None
    if args.brain_graph and Path(args.brain_graph).exists():
        brain_graph = parse_brain_graph_from_json("brain", Path(args.brain_graph).read_text(encoding="utf-8"))
    tasks = generator.generate_tasks_from_gaps(introspections, benchmark_results, brain_graph, max_new_tasks=args.max)
    out_dir = Path(args.output) / f"batch_{len(list(Path(args.output).glob('batch_*')))}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        (out_dir / f"{t.id}.json").write_text(
            json.dumps(
                {
                    "id": t.id,
                    "template_id": t.template_id,
                    "payload": t.payload,
                    "expected_behavior": t.expected_behavior,
                    "labels": t.labels,
                    "difficulty": t.difficulty,
                    "kind": t.kind,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"生成 {len(tasks)} 个任务，保存到 {out_dir}")  # noqa: T201


if __name__ == "__main__":
    main()
