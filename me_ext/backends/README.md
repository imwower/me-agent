# me_ext/backends 扩展点

此目录用于放置真实的多模态 embedding backend 实现，核心规则：

- 不修改 `me_core`，实现一个符合 `EmbeddingBackend` Protocol 的类，如 `RealEmbeddingBackend`，并提供 `create_backend(config)` 工厂函数。
- 可以依赖 torch/clip/第三方 API，但这些依赖不应出现在 `me_core`。
- 在 `AgentConfig` 中设置 `use_dummy_embedding=False` 且 `embedding_backend_module="me_ext.backends.real_backend"` 时，`me_core.alignment.embeddings.create_embedding_backend_from_config` 会尝试通过 import 扩展模块并调用 `create_backend`。
