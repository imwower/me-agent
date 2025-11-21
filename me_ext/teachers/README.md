# me_ext/teachers 扩展点

此目录用于放置调用外部 LLM 或规则系统的 Teacher 实现：

- 实现 `Teacher` Protocol（`generate_advice(TeacherInput) -> TeacherOutput`），可依赖任意第三方库/API。
- 提供可选的工厂或直接在 `scripts/run_evolution_with_teachers.py` / `TeacherManager` 初始化时 import 使用。
- 核心数据结构在 `me_core.teachers`，扩展实现可以工程化或调用线上服务，`me_core` 仅通过接口交互。
