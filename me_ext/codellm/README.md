# me_ext.codellm

本目录放置可选的“写/改代码”类 LLM 客户端实现，避免给 `me_core` 引入额外依赖。

## CodeLLMClient 占位实现

- 位于 `real_codellm.py`，支持 `mock` / `http` / `cli` 三种模式。
- HTTP 模式默认向 `endpoint` 发送 `POST JSON {"prompt": "...", "max_tokens": ...}` 并读取响应文本。
- CLI 模式通过 `subprocess.Popen` 调用外部命令，将 prompt 写入 stdin，stdout 作为模型输出。

示例：

```python
from me_ext.codellm import CodeLLMClient

client = CodeLLMClient({"mode": "mock", "mock_response": "print('hi')"})
resp = client.complete("请输出代码")
```

未来如果要接入真实的 Code-LLM（如 OpenAI / 本地模型），可在本目录新增实现，并在配置中指定 `mode`、`endpoint` 或 `command`。
