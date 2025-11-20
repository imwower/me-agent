# ROADMAP（R1 多模态对齐阶段）

| 模块 | README 规划目标（摘） | 当前实现 | 主要限制 / 待完善 |
| --- | --- | --- | --- |
| perception | 统一多模态感知（文本/图像/音频预留） | `TextPerception`、`ImagePerception`、`MultiModalPerception`，stub 编码写入 `AgentEvent.payload.embeddings` | 图像/音频仅记录元信息与占位向量，缺少真实特征提取与错误恢复策略 |
| alignment / concept space | 跨模态概念对齐与检索 | `alignment/ConceptSpace` + `MultimodalAligner` + `DummyEmbeddingBackend` | 仅 hash 伪向量，无真实模型；阈值/聚类策略简单；缺少持久化与可视化 |
| world_model | 用事件流维护世界知识 | `SimpleWorldModel` 统计事件/工具/概念频次与模态分布 | 无因果/时序建模；概念统计较粗糙；查询接口有限 |
| self_model | 自我描述与能力认知 | `SelfState` + `SimpleSelfModel`，记录模态/能力标签，自述使用中文模板 | 不会主动推理能力缺陷；未结合概念空间总结“我见过什么” |
| drives | 内在驱动力与意图生成 | `SimpleDriveSystem` 回复/时间工具/闲置 + 多模态好奇回复 | 未真正触发外部探索/工具调用；好奇逻辑基于计数，需更精细策略 |
| tools | 统一工具抽象 | `EchoTool`、`TimeTool`、`MultimodalQATool` stub | 多模态 QA 仅占位；无真实模型或外部服务接入 |
| learning | 观察事件并驱动学习 | `SimpleLearner` 计数 + 记录概念模态覆盖，`LearningManager` 管理学习意愿 | 未利用概念向量训练/微调；知识库逻辑极简 |
| dialogue | 将意图转为中文回复 | `RuleBasedDialoguePolicy` 模板式回复，支持提及模态/好奇请求 | 无生成式对话；对齐结果使用有限，缺少多轮记忆 |
| agent | 串联感知→对齐→世界/自我→驱动力→对话/工具 | `SimpleAgent` 将对齐内置于事件写入流程 | 不支持并发/异步；对齐失败无恢复策略；状态持久化最小 |
| demos | CLI 演示/实验脚本 | `demo_cli_agent.py`、`demo_multimodal_cli.py`、char LM/CIFAR demo | Demo 使用占位嵌入，缺少真实多模态模型展示与更丰富案例 |

说明：
- 核心层依赖保持标准库，`DummyEmbeddingBackend` 为占位；真实多模态模型接入需在外部扩展。
- char-LM、CIFAR-100 脚本与核心 Agent 解耦，仅作实验示例。
