# me-agent

> “让我成为我 / 我想 / 我要 / 我做”

me-agent 是一个以“自我驱动的智能体”为核心隐喻的原型系统，用来探索：
- 智能体如何感知世界与自身；
- 如何形成对世界 / 自我的内在模型；
- 如何在内在驱动力的驱动下，持续地“想 / 要 / 做”；
- 如何通过工具、对话、多模态能力，在真实环境中执行行动并反思。

当前仓库已经实现了一套以事件流为核心的最小可运行原型，包括：
- 统一的事件与工具调用类型定义（`AgentEvent` / `ToolCall` / `ToolResult`）；
- 简单的事件流与事件历史工具（`EventStream` / `EventHistory`）；
- 轻量的世界模型、自我模型、驱动力与学习模块；
- 一个可运行的 `SimpleAgent` 循环及命令行 demo 脚本。


## 整体架构概述（文字版）

从高层视角出发，me-agent 将围绕一条循环闭环来设计：

1. **感知（Perception）**  
   - 从外部世界与内部状态中获取信号（文本、环境状态、工具反馈等）。  
   - 将原始信号转化为结构化事件，统一记录为 `AgentEvent`。

2. **世界模型（World Model）**  
   - 基于事件流，维护一个对“外部世界”的抽象表示。  
   - 支持记忆、场景建模、因果关系和简单预测（未来迭代）。

3. **自我模型（Self Model）**  
   - 描述“我是谁”、“我在做什么”、“我能做什么”。  
   - 追踪智能体的状态、能力、偏好、历史行为等。

4. **内在驱动力（Drives）**  
   - 以目标、动机、约束和价值观的形式，给智能体提供“想 / 要”的方向。  
   - 从世界模型 + 自我模型中读取信息，生成新的意图或任务。

5. **工具层（Tools）**  
   - 把具体的外部函数、API、系统操作抽象为工具调用。  
   - 工具调用与结果统一通过 `ToolCall` / `ToolResult` 结构表达，写入事件流。

6. **学习（Learning）**  
   - 从事件流中抽取经验，更新世界模型、自我模型与策略。  
   - 长期目标：实现自我改进与策略调整，而不仅仅是“执行命令”。

7. **对话（Dialogue）**  
   - 处理与人类或其他智能体的自然语言交互。  
   - 将对话输入视为特殊的感知事件，将响应视为行动的一部分。

8. **智能体编排（Agent）**  
   - 调度以上各模块，形成“感知 → 评估 → 决策 → 行动 → 反思”的循环。  
   - 对外暴露统一的接口（如：`step()` / `run()`），让智能体可以嵌入到各种上层系统。

在实现层面上，**事件流（`AgentEvent`）和工具调用结构（`ToolCall` / `ToolResult`）会作为整个系统的“公共语言”**，不同模块通过这些基础类型进行解耦与协作。


## 未来模块规划

以下模块会以 Python 包 / 子包的形式逐步引入，每个模块内部再细化为子组件：

- `perception`  
  - 负责多模态感知（文本、结构化数据、后续可能扩展到图像/音频等）。  
  - 将原始输入规整为统一的事件格式，写入事件流。

- `world_model`  
  - 维护对环境的抽象表示与长期记忆。  
  - 支持查询当前环境状态、历史事件、预测可能的后果。

- `self_model`  
  - 表达“自我”的身份、能力边界、习惯与偏好。  
  - 提供自我反思与自我描述接口，例如“我正在做什么”。

- `drives`  
  - 管理目标与优先级，例如：好奇心、任务完成、信息获取等。  
  - 将外部指令与内部动机统一为“意图”对象，驱动 agent 的行动循环。

- `tools`  
  - 抽象各种外部行动能力（HTTP 调用、本地命令、代码执行等）。  
  - 以 `ToolCall` / `ToolResult` 为核心协议，支持统一的调用与日志记录。

- `learning`  
  - 从事件与工具调用结果中学习策略和偏好。  
  - 包括简单规则更新、参数调整甚至元学习（后续迭代）。

- `dialogue`  
  - 处理自然语言交互，将对话事件融入整体事件流。  
  - 支持多轮对话、角色设定与对话记忆。

- `agent`  
  - 顶层智能体编排器，协调感知、模型、驱动力、工具与学习模块。  
  - 接收输入（环境/用户），输出行动（工具调用/回复/内部状态更新）。


## 当前代码结构概览

当前仓库提供了一套围绕“事件流 + 工具调用”的基础实现，尽量保持简单、可扩展：

- `me_core/`
  - `types.py`：核心通用数据结构，例如 `AgentEvent`、`ToolCall`、`ToolResult` 等；
  - `event_stream.py`：内存中的事件流与事件历史工具；
  - `perception/`：文本/图片感知桩，`TextPerception` 拆分文本事件，`ImagePerception` 记录图片路径与元信息；
  - `alignment/`：概念空间 + Dummy 多模态对齐（R0 占位版，不依赖真实模型）；
  - `world_model/`：`SimpleWorldModel` 维护事件时间线 + 概念记忆，支持按模态/标签查询最近事件；
  - `self_model/`：`SelfState` + `SimpleSelfModel`，记录已见模态、能力标签、最近行动并给出自述；
  - `drives/`：驱动力向量与更新规则，`SimpleDriveSystem` 输出带优先级的意图（回复/调用工具/好奇/自省）；
  - `tools/`：工具协议与注册表，内置 `EchoTool` / `TimeTool` / `HttpGetTool` / `FileReadTool` / `SelfDescribeTool` 等标准库级 stub；
- `learning/`：`SimpleLearner` / `LearningManager`，观察工具成功率与意图结果的轻量学习器；
- `dialogue/`：对话规划器与 `RuleBasedDialoguePolicy`，结合世界/自我/学习信息生成“我想 / 我要 / 我做”式回复；
- `agent/`：`StateStore` / 旧版 `run_once` 主循环，以及新的 `SimpleAgent`（清晰 step 流程、可选 debug/timeline dump）与多 Agent 脚手架。
- `memory/`：`EpisodicMemory` / `SemanticMemory` + Jsonl 存储，支持情节/概念长期记忆；
- `tasks/`：Scenario/TaskStep/TaskResult 定义 + 运行器，便于评估不同场景；
- `introspection/`：生成简单内省日志（总结/错误/改进建议）；
- `config/`：`AgentConfig` 简易配置，支持选择 Dummy 或外部 backend、开关好奇/内省。
- `scripts/`
  - `demo_cli_agent.py`：基于 `SimpleAgent` 的命令行 demo（推荐从这里体验最小闭环）；
  - `demo_multimodal_dummy.py`：使用 Dummy 对齐展示“文本/图片 → 概念空间 → Agent 回复”的占位式多模态 demo；
  - `demo_multi_agent.py`：两个 `SimpleAgent` 轮流对话的多 Agent 示例；
  - `dump_timeline.py` / `view_timeline.py`：读取 JSONL 事件日志并打印时间线；
  - `view_memory.py`：查看持久化的 Episode / 概念记忆；
  - `run_experiments.py`：批量运行 Scenario，输出评估 + 内省日志；
  - 其他脚本：演示自我学习循环、驱动力调整和状态查看等。
- 说明：当前多模态对齐为 R0 Dummy 版本，用于打通结构，后续会接入真实模型。
- `tests/`
  - 覆盖类型定义、感知、驱动力、学习模块以及 `SimpleAgent` 单步行为等。
- `requirements.txt`：当前为空，仅作占位，强调仅依赖 Python 标准库。


## 快速开始

### 准备环境

- 确保使用 Python **3.10+**；
- 不需要安装任何三方依赖（`requirements.txt` 为空）。

在仓库根目录下可以先运行单元测试，确认环境正常：

```bash
python -m unittest
```

### 运行最小对话 demo

在仓库根目录执行：

```bash
python scripts/demo_cli_agent.py
```

你可以在命令行中与 agent 进行多轮简单对话，例如：

- 输入任意句子，观察 agent 如何用“我想 / 我要 / 我做”的风格回应；
- 输入包含“时间”或 `time` 的句子（如“现在几点了？”），agent 会通过 `TimeTool`
  调用内部时间工具，并在回复中体现这一点。

内部大致链路为：

1. 文本输入 → `TextPerception` → 生成 `AgentEvent`（感知事件）；
2. 事件记录到 `EventStream`，同时更新 `SimpleWorldModel` 与 `SimpleSelfModel`；
3. `SimpleDriveSystem` 基于最近事件给出一个 `Intent`（回复 / 调用工具 / 保持安静等）；
4. 如需调用工具，则通过 `ToolCall` / `ToolResult` 连接 `EchoTool` / `TimeTool`；
5. `RuleBasedDialoguePolicy` 结合 Intent 与自我描述生成中文回复；
6. `SimpleLearner` 观察本轮事件，为后续扩展学习逻辑预留接口。

## R2: Memory & Tasks & Introspection

- 长期记忆：`EpisodicMemory`/`SemanticMemory` 支持将事件/概念写入 JSONL，重启可加载。
- 场景评估：在 `me_core/tasks` 定义 Scenario/TaskStep，可用 `scripts/run_experiments.py` 批量运行并生成报告。
- 内省日志：`IntrospectionGenerator` 可基于时间线和工具/意图统计生成“总结/错误/改进”条目。
- 可视化：`scripts/view_timeline.py` 查看事件时间线，`scripts/view_memory.py` 查看持久化记忆。
- 配置与 backend 插拔：`AgentConfig` 支持切换 Dummy/外部 embedding backend，开关好奇/内省等行为。

### 下载 CIFAR-100 数据集（Python 版）

用于 `scripts/train_cifar100_cnn.py` 的示例数据，可直接用仓库脚本下载并解压到 `data/cifar100`：

```bash
python scripts/download_cifar100.py \
  --output data/cifar100 \
  --url https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```

如遇本地证书校验问题，可加 `--insecure`（仅在受信网络使用），如需覆盖已有数据加 `--force`。


## 开发环境与约定

- Python 版本：**3.10+**（使用类型提示与 `dataclasses` 等特性）。
- 依赖：当前阶段仅使用 **Python 标准库**，不引入任何第三方包。
- 类型约定：
  - 所有对外暴露的核心结构（如事件、工具调用）都应有清晰的 `@dataclass` 定义。
  - 尽量使用 `typing` 中的类型标注，以便后续引入静态检查工具（如 `mypy`）。
- 语言约定：
  - 注释与文档（包括本 README）均使用中文，便于聚焦概念本身而非语言障碍。
- 测试：
  - 后续为每个模块增加对应的单元测试；当前仅保留测试目录结构。

未来，当各模块逐步成型后，本 README 会增加：
- 更详细的模块间交互图；
- 典型运行流程示例（例如一个“我想 → 我要 → 我做”的完整回合）；
- 与外部系统集成的示例（CLI / Web / 其他 agent 框架）。


## 基于本地语料的字符级语言模型（实验性）

> 说明：本小节完全基于本地语料和 PyTorch，不依赖 Hugging Face。仅作为“使用大规模中文语料做简单语言建模”的实验性示例，不影响核心 agent 功能。

### 1. 准备本地语料

1）将从 `brightmart/nlp_chinese_corpus` 下载的压缩包放到 `data/` 目录：

- `data/wiki_zh_2019.zip`
- `data/translation2019zh.zip`

2）运行清洗脚本，将原始语料转换为预训练友好的纯文本格式：

```bash
python data/prepare_nlp_chinese_corpus.py
```

运行后会生成：

- `data/wiki_zh_2019/wiki_zh_sentences.txt`：每句话一行，文档间空行；
- `data/translation2019zh/translation2019zh_zh.txt`：每行一个中文句子；
- `data/translation2019zh/translation2019zh_en.txt`：每行一个英文句子。

### 2. 训练字符级语言模型

在项目根目录执行：

```bash
python scripts/train_char_lm_nlp_corpus.py \
  --num-steps 2000 \
  --batch-size 64 \
  --block-size 128 \
  --max-chars-per-file 2000000
```

脚本会：

- 从 `wiki_zh_sentences.txt` 和 `translation2019zh_zh.txt` 加载中文文本；
- 构建字符级词表并训练一个简单的 LSTM 语言模型；
- 定期将 checkpoint 与词表保存到：
  - `outputs/char_lm_nlp_corpus/char_lm_last.pt`
  - `outputs/char_lm_nlp_corpus/vocab.json`

### 3. 交互式生成 Demo

训练完成后，可以启动一个简单的字符级续写 Demo：

```bash
python scripts/demo_char_lm_generate.py
```

进入后：

- 输入任意中文前缀，例如：`经济学是一门研究`
- 模型会基于训练好的字符级语言模型继续生成后续文本；
- 输入 `exit` / `quit` / `q` 或按 `Ctrl+C` / `Ctrl+D` 退出。
