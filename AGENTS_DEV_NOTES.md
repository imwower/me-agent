# me-agent 开发者笔记（现状画像）

> 本文面向协同开发者与未来的“自己”，总结当前仓库的整体结构、已实现能力、明显的桩/原型模块以及与多模态对齐相关的设计现状。  
> 不改变现有 README 的语义，只补充一个更偏工程视角的“现在在哪儿 / 能做啥 / 哪些还没做完”。

## R0 多模态对齐现状速记

- 感知层包含 `TextPerception` 与占位版 `ImagePerception` / `AudioPerception`，默认 demo 仍以文本为主。
- 对齐/概念空间已用 Dummy 向量完成占位实现，等待后续接入真实模型。
- R0 目标：用 Dummy 向量跑通“多模态感知 → Dummy 对齐 → 概念空间 → 世界/自我模型 → Agent”的结构闭环，为后续接入真实模型做准备。

## R0 多模态对齐实现简述

- AgentEvent 新增默认字段：modality/payload/embedding/source/tags，保持向后兼容。
- DummyEmbeddingBackend：基于 sha256 + 伪随机生成稳定单位向量，不依赖真实模型。
- ConceptSpace：维护 ConceptNode（id/name/aliases/centroid/examples），支持余弦最近邻、get_or_create 和观测更新。
- MultimodalAligner：根据事件模态调用 Dummy backend 生成 embedding，对齐/创建概念并回填到事件。
- world_model/self_model：SimpleWorldModel 记录概念计数与模态覆盖；SelfState 追踪 seen_modalities/capability_tags，并提供 describe_self。
- Demo：`python scripts/demo_multimodal_dummy.py [--image <path>]`，展示“文本/图片 → 事件 → Dummy 对齐 → 概念空间 → 世界/自我 → Agent”的闭环。

## 快速状态提示（R1 多模态对齐）

- 核心仍坚持标准库，`requirements.txt` 仅作占位，真实多模态模型需在外部注入。
- 感知/对齐/概念空间为占位实现（Dummy 向量），闭环已在 `SimpleAgent` 中串联，但仍属原型。
- 多模态好奇驱动、自述已初步接通，策略简单，缺少真实模型支撑。
- 更细分的差距表见 `docs/ROADMAP.md`。

## 1. 目录结构与模块职责（简要）

- `me_core/`：核心 Agent 框架
  - `types.py`  
    - 定义统一的数据结构与枚举：`AgentEvent` / `ToolCall` / `ToolResult` / `ToolProgram` / `AgentState` 等。  
    - 已包含多模态占位结构 `MultiModalInput`，用于在感知阶段承载文本、图像、音频、视频的轻量元信息。  
    - 事件来源 `EventSource`、事件语义 `EventKind`、工具类型 `ToolKind` 已集中管理，避免魔法字符串。
  - `event_stream.py`  
    - `EventStream`：负责追加事件、打印日志。  
    - `EventHistory`：维护有限长度的事件窗口，并提供简单统计（按类型计数等）。
  - `perception/`  
    - `BasePerception`：感知接口，`perceive(raw_input) -> AgentEvent`。  
    - `TextPerception`：当前唯一真正使用的感知实现，接收 `str`，包装为 `MultiModalInput(text=...)`，再通过 `encode_to_event` 生成感知事件。  
    - `text_encoder_stub.py` / `image_encoder_stub.py` / `audio_encoder_stub.py` / `video_encoder_stub.py`：多模态编码桩，实现为确定性的伪随机向量。  
    - `processor.py` / `__init__.py`：提供 `encode_to_event` 和 `encode_multimodal`，把 `MultiModalInput` 编码为嵌入字典写入 `AgentEvent.payload`。
  - `world_model/`  
    - `BaseWorldModel`：世界模型接口。  
    - `SimpleWorldModel`：基于 `EventHistory` 的简易实现，只统计事件分布与工具调用成功率，尚未利用多模态向量或概念。
  - `self_model/`  
    - `SelfState`：对“自我”的当前认识（身份、能力、关注主题、局限、最近活动、需要等）。  
    - `BaseSelfModel` / `SimpleSelfModel`：自我模型接口与实现，通过 `update_from_events` 观察事件，聚合一些计数与简单摘要。  
    - `self_summarizer.py`：根据 `SelfState` 生成中文自述。
  - `drives/`  
    - `DriveVector` + `DrivesConfig`：驱动力参数与配置。  
    - `Drive` / `Intent` / `BaseDriveSystem` / `SimpleDriveSystem`：驱动力与意图接口，默认规则：  
      - 最近有人类感知事件 → 意图回复；  
      - 文本中提到“时间”/`time`/“几点” → 意图调用时间工具；  
      - 长时间无事件 → 意图反思；  
      - 否则 idle。
  - `tools/`  
    - 定义 `BaseTool` 接口以及简单工具实现：`EchoTool` / `TimeTool` 等。  
    - `ToolLibrary` 等封装工具注册与调用统计。
  - `learning/`  
    - `BaseLearner` / `SimpleLearner`：学习模块接口与原型实现，目前主要记录观察到的事件，不做复杂参数更新。  
    - `LearningManager`：为未来“主动学习型工具调用”预留接口。
  - `dialogue/`  
    - `BaseDialoguePolicy`：对话策略接口。  
    - `RuleBasedDialoguePolicy`：基于规则的中文回复策略，会使用 `SelfState` 与最近事件历史拼接出“我想 / 我要 / 我做”风格的回复。
  - `agent/`  
    - `BaseAgent`：Agent 抽象基类，定义 `step` 与简单 `run()`。  
    - `SimpleAgent`：完整闭环实现：感知 → 事件流 → 世界模型 / 自我模型更新 → 驱动力决策 → 工具调用（可选）→ 学习 → 对话输出。

- `scripts/`
  - `demo_cli_agent.py`：当前推荐的入口，搭建默认组件并在命令行与 `SimpleAgent` 交互。  
  - `train_char_lm_nlp_corpus.py` / `demo_char_lm_generate.py`：与 Agent 解耦的实验性字符级语言模型训练与生成脚本，使用 `data/` 下的纯文本语料。

- `data/`
  - `prepare_nlp_chinese_corpus.py`：把 `wiki_zh_2019.zip` 和 `translation2019zh.zip` 清洗为预训练友好文本。  
  - `vqa_cn/README.md`：引入中文 SimpleVQA 数据集的说明（当前核心逻辑已不再使用 Hugging Face，数据仅作为参考资源）。  
  - 纯数据文件（zip、txt）不在版本控制中。

- `configs/` / `env/` / `envs/`  
  - 残留了一些早期多任务训练与环境配置文件，现在更多作为参考示例；`env/install.sh` 提供最小依赖安装脚本（以 PyTorch + OCR 工具为主，不再包含 transformers/datasets/open_clip）。

- `tests/`  
  - 覆盖 `me_core` 的核心模块：`test_agent_core.py`, `test_agent_loop.py`, `test_drives.py`, `test_event_stream.py`, `test_perception.py`, `test_self_model.py`, `test_simple_agent.py` 等。  
  - 当前测试主要围绕纯文本 Agent 闭环，尚未覆盖多模态对齐或概念空间相关逻辑（因为这些模块尚未实现）。

## 2. 明显是“桩 / 原型”的模块

- 多模态感知与编码：
  - `MultiModalInput` 只保存原始文本和模态元信息，未引入真实张量或特征。  
  - `text/image/audio/video_encoder_stub.py`：全部是“伪向量”生成器，用于提供固定维度的 embedding，占位而非真正模型。  
  - `encode_multimodal` / `_encode_multimodal_local`：仅简单地把 stub 向量放到 `payload["embeddings"]` 中，没有进一步语义处理。

- 世界模型：
  - `SimpleWorldModel` 只统计事件频次与工具成功率，不利用嵌入、概念或跨模态信息。  
  - 对“世界”的抽象目前基本等价于“事件计数器 + 工具成功率表”。

- 自我模型：
  - `SelfState` 信息丰富，但更新策略（`SelfUpdater`）只是基于事件摘要做简单计数与文本拼接，并未引入向量表征或概念层自我认识。  
  - 没有区分“我会不会看图/听音频”等多模态能力。

- 学习模块：
  - `SimpleLearner` 目前只观察事件并保留基础接口，未真正利用事件流进行优化或记忆更新。  
  - `LearningManager` 的“何时主动学习”逻辑仍是占位。

- 对话模块：
  - `RuleBasedDialoguePolicy` 完全基于规则和模板，没有根据多模态证据或概念空间做回答。  
  - 对于“你会看图吗？”之类的问题，目前只能依靠 SelfState 中的静态描述。

- 多模态 / 训练：
  - 之前存在的 transformers + Hugging Face 多模态训练 pipeline 已被移除，剩余的多任务配置和说明主要作为未来扩展参考。  
  - 当前项目不再依赖 Hugging Face，所有多模态能力都需要在轻量级本地模块（或外部扩展）上重新实现。

## 3. 与多模态对齐相关的现有设计

- 多模态输入与事件结构：
  - `MultiModalInput`：已经为文本、图像、音频、视频预留字段，但只有 `TextPerception` 在实际使用。  
  - `encode_to_event`：会把多模态编码结果写入 `AgentEvent.payload["embeddings"]`，这是接入“概念空间/对齐模块”的天然锚点。  
  - `AgentEvent.payload["raw"]` 中保存原始文本和各模态的元信息，便于 later 构建 `ImageRef` / `AudioRef` 等结构。

- 感知 → 事件流 → 世界/自我模型：
- `SimpleAgent` 通过 `_register_event` 集中完成“写 event_stream + world_model/self_model 更新”，后续只要让 world/self model 利用 embedding 和概念空间即可。  
  - `SimpleDriveSystem` 的决策接口已经接收 `world_model` 和 `self_model` 引用，为根据“概念/模态统计”调整意图留有余地。

这些设计说明：**多模态对齐子系统可以自然地插在“感知到的 embedding → 概念空间 → 世界/自我模型更新”这条链路中，不需要大改 Agent 主循环的结构。**

## 4. 潜在坑与需要注意的地方

- 类型与循环依赖：
  - `me_core.types` 已经在 try/except 中懒加载 `SelfState` 与 `DriveVector`，新类型扩展时要避免增加运行时循环依赖。  
  - 感知模块与世界/自我模型之间尽量只通过 `AgentEvent` + 简单数据结构交互，不直接互相引用类。

- 事件 payload 结构：
  - 目前 `AgentEvent.payload` 是一个松散的 `dict[str, Any]`，各模块约定的键名略有不同（例如 `"raw"`、`"embeddings"`、`"kind"`）。  
  - 在设计概念空间/对齐模块时，需要明确采用哪些键，并尽量在一个地方集中约定（例如 alignment 模块的 docstring 或常量），防止后期“魔法字段”泛滥。

- 多模态 stub 与真实模型分层：
  - 现在的 encoder_stub 在 `me_core` 里，意味着 core 层默认只依赖标准库和简单 hash/随机数逻辑。  
  - 未来接入真实多模态模型（如 CLIP、中文多模态 LLM）时，建议放在 `src/` 或独立扩展包中，通过接口（如 `EmbeddingBackend`）注入，而不是直接改动 core。

- 测试覆盖：
  - 现有单元测试主要覆盖纯文本流程，增加多模态和对齐模块后，需要补充新的测试用例，同时确保旧测试仍通过，避免破坏已有行为。  
  - 尤其是 `AgentEvent` / `SimpleAgent` 的接口，应保持向后兼容：新增字段要有默认值，`to_dict/from_dict` 要容忍缺失。

- 日志与可观测性：
  - 目前 demo 路径（`demo_cli_agent.py` + `SimpleAgent.step`）在终端中打印了大量中文日志（感知/驱动力/工具/对话）。  
  - 新增多模态对齐与概念空间时，建议继续沿用这一风格，在关键节点打印一两行易读的中文日志，方便人类开发者理解系统行为。

---

后续改造计划（简要）：

1. 在 `AgentEvent` 与 `types.py` 中补充多模态相关字段（modality/source/tags 等），并引入 `ImageRef` / `AudioRef` 等轻量结构，为 alignment 做准备。  
2. 在 `me_core/alignment/` 下实现 `ConceptNode` / `ConceptSpace` / `DummyEmbeddingBackend` / `MultimodalAligner`，形成“概念空间 + 对齐器”骨架。  
3. 扩展 `SimpleWorldModel` 与 `SimpleSelfModel`，让它们能够消费概念空间信息（例如概念出现频率、多模态分布、自我能力标签）。  
4. 在 `SimpleDriveSystem` / `SimpleLearner` / `RuleBasedDialoguePolicy` 中引入对对齐结果和概念统计的简单使用（好奇驱动、自我描述、多模态问答）。  
5. 增加 `scripts/demo_multimodal_cli.py`，以文本 + 图像路径为输入，跑通“感知 → 对齐 → 世界/自我更新 → 驱动力 → 对话”的最小多模态闭环。  
6. 为新模块补充单元测试，并保持原有测试全部通过。
