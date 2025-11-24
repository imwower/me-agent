from __future__ import annotations

import http.client
import json
import logging
import subprocess
from typing import List
from urllib.parse import urlparse

from me_core.teachers.interface import Teacher
from me_core.teachers.types import ConfigPatch, PolicyPatch, TeacherInput, TeacherOutput

logger = logging.getLogger(__name__)


class RealTeacher(Teacher):
    """
    基于外部 LLM/规则系统的 Teacher 占位实现。
    支持两种调用模式：
    - http: 通过 http.client 请求一个返回文本/JSON 的 API
    - cli: 通过 subprocess 调用外部命令，stdin 输入 prompt，stdout 读取回复
    """

    name = "real_teacher"

    def __init__(self, config: dict) -> None:
        self.config = config or {}

    def _call_http_llm(self, prompt: str) -> str:
        url = self.config.get("endpoint") or ""
        parsed = urlparse(url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=10)
        path = parsed.path or "/"
        try:
            body = json.dumps({"prompt": prompt}).encode("utf-8")
            conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read().decode("utf-8", errors="ignore")
            return data
        except Exception as exc:  # pragma: no cover - 依赖外部服务
            logger.warning("RealTeacher http 调用失败: %s", exc)
            return '{"advice": "fallback", "patches": []}'
        finally:
            conn.close()

    def _call_cli_llm(self, prompt: str) -> str:
        cmd = self.config.get("command") or "cat"
        try:
            proc = subprocess.Popen(
                cmd.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _ = proc.communicate(prompt.encode("utf-8"), timeout=10)
            return stdout.decode("utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - 依赖外部命令
            logger.warning("RealTeacher cli 调用失败: %s", exc)
            return '{"advice": "fallback", "patches": []}'

    def _build_prompt(self, ti: TeacherInput) -> str:
        data = {
            "scenario_id": ti.scenario_id,
            "introspection": ti.introspection.to_dict() if ti.introspection else None,
            "current_config": ti.current_config,
            "notes": ti.notes,
            "experiment_results": [
                {
                    "repo_id": r.step.repo_id,
                    "kind": r.step.kind,
                    "metrics": r.metrics,
                    "returncode": r.returncode,
                }
                for r in (ti.experiment_results or [])
            ],
            "brain_graph": ti.brain_graph.summary() if ti.brain_graph else None,
        }
        prompt = (
            "你是一个策略老师，请根据以下 JSON 给出改进建议，并返回 JSON，其中包含 advice 文本和 patches。"
            "patches 是对象数组，每个包含 target/path/value/reason。\n"
            f"{json.dumps(data, ensure_ascii=False)}\n"
            "请仅输出 JSON，例如：{\"advice\": \"...\", \"patches\": [{\"target\": \"drives\", \"path\": \"curiosity.min_concept_count\", \"value\": 2, \"reason\": \"更多探索\"}]}"
        )
        return prompt

    def _parse_patches_from_response(self, response: str) -> tuple[List[PolicyPatch], List[ConfigPatch]]:
        patches: List[PolicyPatch] = []
        config_patches: List[ConfigPatch] = []
        try:
            obj = json.loads(response)
            patch_items = obj.get("patches") or []
            for item in patch_items:
                if not isinstance(item, dict):
                    continue
                patches.append(
                    PolicyPatch(
                        target=str(item.get("target") or "drives"),
                        path=str(item.get("path") or ""),
                        value=item.get("value"),
                        reason=str(item.get("reason") or ""),
                    )
                )
            cfg_items = obj.get("config_patches") or []
            for item in cfg_items:
                if not isinstance(item, dict):
                    continue
                config_patches.append(
                    ConfigPatch(
                        repo_id=str(item.get("repo_id") or ""),
                        config_path=str(item.get("config_path") or ""),
                        path=str(item.get("path") or ""),
                        value=item.get("value"),
                        reason=str(item.get("reason") or ""),
                    )
                )
        except Exception:
            logger.warning("解析 Teacher 响应失败，返回空补丁。")
        return patches, config_patches

    def generate_advice(self, ti: TeacherInput) -> TeacherOutput:
        prompt = self._build_prompt(ti)
        mode = self.config.get("mode") or "cli"
        if mode == "http":
            raw = self._call_http_llm(prompt)
        else:
            raw = self._call_cli_llm(prompt)
        patches, config_patches = self._parse_patches_from_response(raw)
        return TeacherOutput(
            advice_text=raw,
            policy_patches=patches,
            config_patches=config_patches,
            meta={"raw_response": raw, "mode": mode},
        )


def create_teacher(config_dict: dict) -> Teacher:
    return RealTeacher(config_dict or {})
