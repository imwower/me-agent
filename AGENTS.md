# Repository Guidelines

## Project Structure & Module Organization

- `me_core/`: core agent loop, perception, drives, self/world models, tools, dialogue, learning.
- `src/`: multimodal training, data modules, models, inference and evaluation utilities.
- `scripts/`: CLI demos, data download/prepare scripts, training and evaluation entrypoints.
- `configs/`: YAML configs for datasets, models, and training runs.
- `tests/`: unit tests for `me_core` and key `src` modules.
- `data/`, `envs/`, `env/`: local data prep helpers and example environments (not committed to CI).

## Build, Test, and Development Commands

- Run full test suite: `python -m unittest`.
- Run a single test module: `python -m unittest tests.test_agent_core`.
- CLI demo (recommended first entry): `python scripts/demo_cli_agent.py`.
- Other useful scripts: `python scripts/run_agent_demo.py`, `python scripts/run_agent_interactive.py`, `bash scripts/train_vqa_cn.sh` (and similar `train_*` / `eval_*` scripts).

## Coding Style & Naming Conventions

- Python 3.10+, PEP 8 style, 4-space indentation, type hints everywhere public.
- Modules and packages: `snake_case`; classes: `PascalCase`; functions/variables: `snake_case`.
- Use `@dataclass` for core data structures (`AgentEvent`, `ToolCall`, etc.).
- Prefer Chinese for user-facing docs, comments, and messages; keep code identifiers in English.
- Avoid new runtime dependencies unless clearly justified and documented.

## Testing Guidelines

- Add or update `unittest`-style tests under `tests/` for new behavior.
- Name tests after the feature under test (e.g., `test_simple_agent.py`, `test_world_model.py`).
- Ensure `python -m unittest` passes before opening a PR; add focused tests for regressions.

## Commit & Pull Request Guidelines

- Follow existing history: `feat(scope): 描述`, `fix(scope): 描述`, or a clear English summary (e.g., `Add multitask model training and inference pipeline`).
- Keep commits small and logically scoped; include relevant configs and tests.
- PRs should describe motivation, main changes, and how to reproduce tests or demos; link issues when applicable and include screenshots or logs for user-visible changes.

## Agent-Specific Instructions

- When editing `me_core/`, also follow `me_core/AGENTS.md` (no side effects on import, reasonable defaults, backward-compatible state).
- Preserve existing APIs and behavior unless the change is explicitly requested; otherwise add new, opt-in paths.
- Do not introduce external services or network calls into core modules or tests.
