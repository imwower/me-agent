import logging
import unittest

from me_core.drives.drive_vector import DriveVector
from me_core.learning.learning_manager import LearningManager
from me_core.tools.registry import ToolInfo, ToolRegistry

# 为测试输出配置基础日志，便于观察学习过程
logging.basicConfig(level=logging.INFO)


class LearningManagerTestCase(unittest.TestCase):
    """学习管理器的行为测试。"""

    def _build_registry_with_search(self) -> ToolRegistry:
        """构造一个包含 search_papers 工具的注册表。"""

        registry = ToolRegistry()
        registry.register_tool(
            ToolInfo(
                name="search_papers",
                type="knowledge",
                cost=0.3,
                description="检索与给定主题相关的论文与资料（桩实现）。",
                good_for=["paper", "论文", "SNN"],
            )
        )
        return registry

    def test_learning_desire_and_active_learning(self) -> None:
        """高不确定性 + 高好奇心时，应主动调用学习工具。"""

        registry = self._build_registry_with_search()
        manager = LearningManager(registry=registry)

        # 构造一个好奇心和学习强度都较高的驱动力向量
        drives = DriveVector(
            chat_level=0.5,
            curiosity_level=0.9,
            exploration_level=0.6,
            learning_intensity=0.9,
            social_need=0.4,
            data_need=0.7,
        )
        uncertainty = 0.9

        desire = manager.compute_learning_desire(uncertainty, drives)
        self.assertGreaterEqual(desire, 0.2)

        context = {"topic": "延迟记忆 SNN 模型"}
        results = manager.maybe_learn(uncertainty, drives, context)

        # 学习意愿足够高时，至少应调用一个工具
        self.assertGreaterEqual(len(results), 1)
        # 成功结果应写入知识库
        self.assertGreaterEqual(len(manager.knowledge_base), 1)
        self.assertIn("tool_name", manager.knowledge_base[0])
        self.assertIn("topic", manager.knowledge_base[0])
        self.assertIn("summary", manager.knowledge_base[0])

    def test_learning_desire_too_low_no_learning(self) -> None:
        """学习意愿过低时，不应调用任何学习工具。"""

        registry = self._build_registry_with_search()
        manager = LearningManager(registry=registry)

        # 好奇心和学习强度都较低，不确定性也较低
        drives = DriveVector(
            chat_level=0.5,
            curiosity_level=0.1,
            exploration_level=0.2,
            learning_intensity=0.1,
            social_need=0.3,
            data_need=0.2,
        )
        uncertainty = 0.1

        desire = manager.compute_learning_desire(uncertainty, drives)
        self.assertLess(desire, 0.2)

        context = {"topic": "任意主题"}
        before_size = len(manager.knowledge_base)
        results = manager.maybe_learn(uncertainty, drives, context)

        self.assertEqual(len(results), 0)
        self.assertEqual(len(manager.knowledge_base), before_size)

    def test_fallback_to_all_tools_when_no_topic_match(self) -> None:
        """当主题与任何工具关键词都不匹配时，应退回到使用全部工具。"""

        registry = ToolRegistry()
        # 构造一个只对特定关键词敏感的工具，并使用与之完全无关的 topic
        registry.register_tool(
            ToolInfo(
                name="generic_tool",
                type="knowledge",
                cost=0.2,
                description="通用学习工具（用于测试回退逻辑）。",
                good_for=["完全不同的关键词"],
            )
        )

        manager = LearningManager(registry=registry)

        drives = DriveVector(
            chat_level=0.5,
            curiosity_level=0.9,
            exploration_level=0.9,
            learning_intensity=0.9,
            social_need=0.4,
            data_need=0.9,
        )
        uncertainty = 0.9

        context = {"topic": "与工具关键词完全不匹配的主题"}
        results = manager.maybe_learn(uncertainty, drives, context)

        # 尽管 topic 与 good_for 不匹配，也应通过“回退为全部工具”机制触发至少一次调用
        self.assertGreaterEqual(len(results), 1)

    def test_query_knowledge_and_max_entries(self) -> None:
        """query_knowledge 应按主题返回最近记录，并对知识库长度进行控制。"""

        registry = self._build_registry_with_search()
        manager = LearningManager(registry=registry, max_knowledge_entries=5)

        # 手动追加多条学习记录，部分主题相关，部分无关
        for i in range(10):
            entry = {
                "tool_name": "search_papers",
                "topic": f"topic-{i}",
                "summary": f"关于 topic-{i} 的学习记录",
                "details": {},
            }
            manager.add_knowledge_entry(entry)

        # 知识库长度应被截断到 max_knowledge_entries
        self.assertLessEqual(len(manager.knowledge_base), 5)

        # 查询一个在最近几条中出现的主题，应有命中
        results = manager.query_knowledge("topic-9", max_results=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(any("topic-9" in r["topic"] for r in results))


if __name__ == "__main__":
    unittest.main()
