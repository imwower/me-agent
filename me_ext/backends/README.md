# me_ext/backends 扩展点

此目录用于放置真实的多模态 embedding backend 实现，核心规则：

- 不修改 `me_core`，实现一个符合 `EmbeddingBackend` Protocol 的类，如 `RealEmbeddingBackend`，并提供 `create_backend(config)` 工厂函数。
- 可以依赖 torch/clip/第三方 API，但这些依赖不应出现在 `me_core`。
- 在 `AgentConfig` 中设置 `use_dummy_embedding=False` 且 `embedding_backend_module="me_ext.backends.real_backend"` 时，`me_core.alignment.embeddings.create_embedding_backend_from_config` 会尝试通过 import 扩展模块并调用 `create_backend`。

示例模板（见 `real_backend.py`）：

```python
from typing import List
from me_core.alignment.embeddings import EmbeddingBackend
from me_core.types import ImageRef, AudioRef

class RealEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_config: dict) -> None:
        # 在这里加载真实模型（CLIP/多模态 embedding），可用 torch/onnx/http 等
        self.model_config = model_config

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        # 返回 L2 归一化的浮点向量
        ...

    def embed_image(self, image_refs: List[ImageRef]) -> List[List[float]]:
        # 根据 image_ref.path 读取图片并编码
        ...

    def embed_audio(self, audio_refs: List[AudioRef]) -> List[List[float]]:
        ...

def create_backend(config: dict) -> EmbeddingBackend:
    return RealEmbeddingBackend(config)
```

配置示例（AgentConfig）：

```json
{
  "use_dummy_embedding": false,
  "embedding_backend_module": "me_ext.backends.real_backend",
  "embedding_backend_kwargs": {
    "model_name": "your_clip_model",
    "device": "cuda:0"
  }
}
```
