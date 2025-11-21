# me_ext/teachers 扩展点

此目录用于放置调用外部 LLM 或规则系统的 Teacher 实现：

- 实现 `Teacher` Protocol（`generate_advice(TeacherInput) -> TeacherOutput`），可依赖任意第三方库/API。
- 提供可选的工厂或直接在 `scripts/run_evolution_with_teachers.py` / `TeacherManager` 初始化时 import 使用。
- 核心数据结构在 `me_core.teachers`，扩展实现可以工程化或调用线上服务，`me_core` 仅通过接口交互。

## 示例：real_teacher.py

占位实现支持两种调用方式：

```python
teacher = RealTeacher({
  "mode": "http",
  "endpoint": "http://localhost:8080/llm"
})
# 或 mode=cli, command="python your_llm_cli.py"
```

RealTeacher 会构造 prompt（包含场景 id、内省日志、当前策略），要求外部 LLM 返回 JSON：

```json
{
  "advice": "多探索",
  "patches": [
    {"target": "drives", "path": "curiosity.min_concept_count", "value": 2, "reason": "鼓励探索"}
  ]
}
```

将 `teacher_module` 配置为 `me_ext.teachers.real_teacher` 即可通过 `create_teacher_manager_from_config` 加载。
