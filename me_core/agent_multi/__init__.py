from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from me_core.agent.simple_agent import SimpleAgent


@dataclass
class AgentShell:
    id: str
    agent: SimpleAgent


class ConversationHub:
    """
    管理多个 AgentShell 之间的对话/博弈。
    """

    def __init__(self, agents: List[AgentShell]) -> None:
        self._agents: Dict[str, AgentShell] = {agent.id: agent for agent in agents}

    def run_turn(self, speaker_id: str, message: str) -> Dict[str, str]:
        """
        将 message 送给指定 agent，拿到它的回复；
        并可广播/传递给其他 agent（当前策略为简单广播）。
        返回形如 {agent_id: reply_text} 的字典。
        """

        responses: Dict[str, str] = {}
        speaker = self._agents.get(speaker_id)
        if speaker is None:
            return responses

        reply = speaker.agent.step(message) or ""
        responses[speaker_id] = reply

        for other_id, shell in self._agents.items():
            if other_id == speaker_id:
                continue
            forwarded = f"{speaker_id} 说：{reply or message}"
            responses[other_id] = shell.agent.step(forwarded) or ""

        return responses


__all__ = ["AgentShell", "ConversationHub"]
