from __future__ import annotations

import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.models.bridges.qformer import SimpleQFormerBridge
from src.models.heads.answerability_head import AnswerabilityHead
from src.models.heads.chart_head import ChartHead
from src.models.heads.ocr_pointer_head import OCRPointerHead
from src.models.heads.vqa_head import VQAHead
from src.models.vision.openclip_vit_b16 import OpenCLIPVisionEncoder

logger = logging.getLogger(__name__)


class MultiTaskModel(nn.Module):
    """多任务多模态模型骨架。

    设计目标：
        - 将视觉编码器、桥接层、文本解码器与多任务 head 统一封装；
        - 支持 VQA / OCR-VQA / Chart-QA 等任务共享一套 backbone；
        - 当前实现主要作为结构骨架，训练时仍可仅使用文本侧损失，
          后续逐步启用视觉与 pointer 相关分支。

    注意：
        - 为了避免在导入模块时就下载大模型，建议仅在训练脚本中实例化；
        - 本类不处理优化器与调度器，这些由 training.optimizer 管理。
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device

        model_cfg = model_cfg.get("model", {})

        # 1) 视觉编码器（OpenCLIP），默认冻结，仅用于特征提取
        vision_cfg = model_cfg.get("vision_encoder", {}) or {}
        vision_name = vision_cfg.get("name", "open_clip_vit_b16")
        if vision_name != "open_clip_vit_b16":
            logger.warning("当前 MultiTaskModel 仅针对 open_clip_vit_b16 做了封装。")
        vision_pretrained = vision_cfg.get("pretrained", "openai")
        vision_embed_dim = int(vision_cfg.get("embed_dim", 512))

        self.vision_encoder = OpenCLIPVisionEncoder(
            model_name="ViT-B-16",
            pretrained=vision_pretrained,
            device=str(device),
        )
        self.vision_dim = vision_embed_dim

        # 2) Q-Former 风格桥接层
        bridge_cfg = model_cfg.get("bridge", {}) or {}
        bridge_hidden = int(bridge_cfg.get("hidden_dim", 512))
        bridge_num_queries = int(bridge_cfg.get("num_query_tokens", 32))
        bridge_use_lora = bool(bridge_cfg.get("use_lora", False))
        bridge_lora_r = int(bridge_cfg.get("lora_r", 8))
        bridge_lora_alpha = float(bridge_cfg.get("lora_alpha", 16.0))

        self.bridge = SimpleQFormerBridge(
            vision_dim=self.vision_dim,
            hidden_dim=bridge_hidden,
            num_query_tokens=bridge_num_queries,
            use_lora=bridge_use_lora,
            lora_r=bridge_lora_r,
            lora_alpha=bridge_lora_alpha,
        )

        # 3) 文本解码器（当前使用 AutoModelForCausalLM）
        text_cfg = model_cfg.get("text_decoder", {}) or {}
        text_model_name = text_cfg.get("model_name", "gpt2")
        self.max_length = int(text_cfg.get("max_length", 64))

        logger.info("MultiTaskModel: 加载文本模型 %s", text_model_name)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(text_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.text_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(text_model_name)
        self.text_model.to(self.device)

        hidden_dim = self.text_model.config.hidden_size  # type: ignore[attr-defined]
        vocab_size = self.text_model.config.vocab_size  # type: ignore[attr-defined]

        # 4) 多任务头部
        heads_cfg = model_cfg.get("heads", {}) or {}
        vqa_hidden = int(heads_cfg.get("vqa", {}).get("hidden_dim", hidden_dim))
        # 可答性头的输入维度需要与文本隐藏维度一致，避免线性层维度不匹配
        ans_hidden = hidden_dim
        ocr_hidden = int(heads_cfg.get("ocr_pointer", {}).get("hidden_dim", hidden_dim // 2))
        chart_hidden = int(heads_cfg.get("chart", {}).get("hidden_dim", hidden_dim // 2))

        self.vqa_head = VQAHead(hidden_dim=vqa_hidden, vocab_size=vocab_size)
        self.answerability_head = AnswerabilityHead(hidden_dim=ans_hidden)
        self.ocr_head = OCRPointerHead(hidden_dim=ocr_hidden)
        self.chart_head = ChartHead(hidden_dim=chart_hidden)

        # OCR 与图表元素的哈希嵌入，用于在不依赖大模型编码的情况下
        # 为指针网络提供可训练的 token/元素表征。
        ocr_vocab_size = int(model_cfg.get("ocr_token_vocab_size", 4096))
        chart_vocab_size = int(model_cfg.get("chart_elem_vocab_size", 4096))
        self.ocr_token_emb = nn.Embedding(ocr_vocab_size, hidden_dim)
        self.chart_elem_emb = nn.Embedding(chart_vocab_size, hidden_dim)

        logger.info(
            "MultiTaskModel 初始化完成: vision=%s, text=%s, hidden_dim=%d",
            vision_name,
            text_model_name,
            hidden_dim,
        )

        # 默认冻结大规模 backbone（视觉编码器与文本模型），
        # 仅通过桥接层 LoRA 与任务头进行轻量微调。
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.text_model.parameters():
            p.requires_grad = False

    # ====================== 文本侧辅助接口 ======================

    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """将文本批次编码为输入张量。"""

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def lm_forward(self, texts: List[str]) -> torch.Tensor:
        """对一批文本执行自回归语言模型前向，并返回 loss。"""

        enc = self.encode_texts(texts)
        outputs = self.text_model(**enc, labels=enc["input_ids"])
        return outputs.loss  # type: ignore[no-any-return]

    def forward_with_answerability(
        self,
        texts: List[str],
        answerable_labels: List[Optional[bool]],
        images: Optional[List["object"]] = None,
        answerability_weight: float = 1.0,
    ) -> torch.Tensor:
        """联合计算文本 LM 损失与可答性二分类损失。

        说明：
            - 文本部分使用标准自回归 loss；
            - 可答性部分使用 AnswerabilityHead，对隐藏状态做二分类，
              仅对具有显式 answerable 标签的样本计算 BCE；
            - 若提供 images 且与 texts 数量一致，则通过桥接层获得一个
              简单的视觉摘要，并与文本表征相加，形成多模态表征。
        """

        enc = self.encode_texts(texts)
        # 要同时拿到 loss 与隐藏状态，因此开启 output_hidden_states
        outputs = self.text_model(
            **enc,
            labels=enc["input_ids"],
            output_hidden_states=True,
        )
        lm_loss: torch.Tensor = outputs.loss  # type: ignore[assignment]

        # 若没有任何样本带有 answerable 标签，则只返回 LM 损失
        if not any(label is not None for label in answerable_labels):
            return lm_loss

        hidden_states = outputs.hidden_states[-1]  # type: ignore[index]
        # 使用序列最后一个 token 的隐藏状态作为文本表征
        pooled_text = hidden_states[:, -1, :]  # (batch, hidden)

        # 尝试引入视觉信息：通过桥接层生成 query token，并做简单平均作为视觉摘要
        pooled = pooled_text
        if images is not None and len(images) == len(texts):
            try:
                vision_feats = self.encode_images(images)  # (batch, vision_dim)
                bridge_out = self.bridge_vision(vision_feats)  # (batch, num_q, hidden_dim)
                pooled_vision = bridge_out.mean(dim=1)
                # 若维度匹配，则将视觉摘要与文本表征相加，形成多模态表示
                if pooled_vision.shape == pooled_text.shape:
                    pooled = pooled_text + pooled_vision
            except Exception:
                # 若视觉路径出错，不影响文本训练
                pooled = pooled_text

        logits = self.answerability_head(pooled).view(-1)

        # 构造标签与掩码
        labels_list = []
        mask_list = []
        for label in answerable_labels:
            if label is None:
                labels_list.append(0.0)
                mask_list.append(0.0)
            else:
                labels_list.append(1.0 if label else 0.0)
                mask_list.append(1.0)

        target = torch.tensor(labels_list, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask_list, dtype=torch.float32, device=self.device)

        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        per_example = bce(logits, target)

        # 仅对 mask=1 的样本取平均
        denom = mask.sum().clamp_min(1.0)
        answerability_loss = (per_example * mask).sum() / denom

        return lm_loss + answerability_weight * answerability_loss

    # ====================== 视觉侧辅助接口 ======================

    def encode_images(self, pil_images: List["object"]) -> torch.Tensor:
        """使用 OpenCLIP 对图像进行编码，返回 (batch, vision_dim) 特征。"""

        with torch.no_grad():
            feats = self.vision_encoder.encode_pil_images(pil_images)
        return feats

    def bridge_vision(self, vision_feats: torch.Tensor) -> torch.Tensor:
        """将视觉特征通过桥接层投影到 query token 空间。"""

        return self.bridge(vision_feats)

    # ====================== OCR / 图表辅助编码 ======================

    @staticmethod
    def _hash_to_bucket(text: str, num_buckets: int) -> int:
        """将任意文本哈希到 [0, num_buckets) 区间。

        说明：
            - 使用 md5 保证跨进程/多次运行的一致性；
            - 仅作为简易的离散 ID，与语义无强约束。
        """

        if num_buckets <= 0:
            return 0
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(h[:8], 16) % num_buckets

    def encode_ocr_tokens(
        self,
        batch_ocr_tokens: List[List[Dict[str, Any]]],
    ) -> torch.Tensor:
        """将一批样本的 OCR token 文本编码为隐藏向量。

        实现方式：
            - 对每个 token 文本做哈希，映射到一个离散 bucket；
            - 使用可训练的 nn.Embedding 查表得到向量；
            - 对不同样本长度进行 padding，空位使用 0 向量。
        """

        batch_size = len(batch_ocr_tokens)
        hidden_dim = self.text_model.config.hidden_size  # type: ignore[attr-defined]
        max_tokens = max((len(toks) for toks in batch_ocr_tokens), default=1)

        ocr_hidden = torch.zeros(
            batch_size,
            max_tokens,
            hidden_dim,
            device=self.device,
        )
        num_buckets = self.ocr_token_emb.num_embeddings

        for i, toks in enumerate(batch_ocr_tokens):
            for j, tok in enumerate(toks):
                if j >= max_tokens:
                    break
                text = str(tok.get("text") or "")
                idx = self._hash_to_bucket(text, num_buckets)
                ocr_hidden[i, j] = self.ocr_token_emb.weight[idx]

        return ocr_hidden

    def encode_chart_elements(
        self,
        batch_chart_elements: List[List[Dict[str, Any]]],
    ) -> torch.Tensor:
        """将一批样本的图表元素编码为隐藏向量。

        实现方式：
            - 为每个元素提取一个代表性文本（如 label/x/name），若缺失则退化为 id；
            - 将文本哈希到离散 bucket，并通过 nn.Embedding 查表得到向量；
            - 对不同样本元素个数进行 padding，空位使用 0 向量。
        """

        batch_size = len(batch_chart_elements)
        hidden_dim = self.text_model.config.hidden_size  # type: ignore[attr-defined]
        max_elems = max((len(elems) for elems in batch_chart_elements), default=1)

        chart_hidden = torch.zeros(
            batch_size,
            max_elems,
            hidden_dim,
            device=self.device,
        )
        num_buckets = self.chart_elem_emb.num_embeddings

        for i, elems in enumerate(batch_chart_elements):
            for j, elem in enumerate(elems):
                if j >= max_elems:
                    break
                meta = elem.get("meta") or {}
                # 选取若干常见字段作为“名称”，不足时回退到 id
                label = (
                    str(meta.get("label") or "")
                    or str(meta.get("x") or "")
                    or str(meta.get("name") or "")
                    or str(meta.get("category") or "")
                    or str(elem.get("id") or "")
                )
                idx = self._hash_to_bucket(label, num_buckets)
                chart_hidden[i, j] = self.chart_elem_emb.weight[idx]

        return chart_hidden

    # ====================== OCR 指针式损失（轻量实战化） ======================

    def compute_ocr_pointer_loss_from_samples(
        self,
        samples: List[Dict[str, Any]],
        ignore_index: int = -100,
    ) -> Optional[torch.Tensor]:
        """根据一批 OCR-VQA 样本，计算简单的指针式拷贝损失。

        思路：
            - 从样本中提取 evidence.ocr_tokens 作为候选 token 池；
            - 将 token 文本通过哈希嵌入映射为向量；
            - 对于每条样本，若某个 token 文本是答案字符串的子串，
              则将其索引作为监督信号；否则忽略该样本；
            - 使用 OCRPointerHead 进行分类，并通过 CrossEntropyLoss 训练。

        注意：
            - 当前版本尚未显式引入问题文本/视觉特征，更多是从“答案包含的 token”
              这一弱监督中学习一个“偏向正确 token 的指针分布”；
            - 后续可以在 decoder_hidden 中引入问题编码，使指针更加依赖问题语境。
        """

        batch_ocr_tokens: List[List[Dict[str, Any]]] = []
        answers: List[str] = []
        target_indices: List[int] = []

        for ex in samples:
            ev = ex.get("evidence") or {}
            toks = ev.get("ocr_tokens") or []
            batch_ocr_tokens.append(toks)

            ans_list = ex.get("answers") or []
            ans = str(ans_list[0]) if ans_list else ""
            answers.append(ans)

            target = ignore_index
            if ans:
                for idx, tok in enumerate(toks):
                    text = str(tok.get("text") or "")
                    if text and text in ans:
                        target = idx
                        break
            target_indices.append(target)

        if not any(t != ignore_index for t in target_indices):
            # 没有任何样本提供有效指针监督，则不计算该损失
            return None

        ocr_hidden = self.encode_ocr_tokens(batch_ocr_tokens)  # (B, N, H)
        batch_size, _, hidden_dim = ocr_hidden.shape

        # 简单使用一个全局可训练查询向量作为 decoder_hidden 占位，
        # 后续可以替换为基于问题文本/多模态的编码。
        if not hasattr(self, "ocr_query"):
            self.ocr_query = nn.Parameter(torch.zeros(hidden_dim, device=self.device))
            nn.init.normal_(self.ocr_query, mean=0.0, std=0.02)

        decoder_hidden = self.ocr_query.unsqueeze(0).expand(batch_size, -1)  # (B, H)
        pointer_logits = self.ocr_head(decoder_hidden, ocr_hidden)  # (B, N)

        target_tensor = torch.tensor(
            target_indices,
            dtype=torch.long,
            device=self.device,
        )
        loss = self.ocr_head.compute_loss(pointer_logits, target_tensor, ignore_index=ignore_index)
        return loss

    # ====================== 图表值抽取损失（轻量实战化） ======================

    @staticmethod
    def _extract_numeric_from_meta(meta: Dict[str, Any]) -> Optional[float]:
        """从图表元素的 meta 字段中尝试抽取一个数值。

        规则：
            - 优先从常见字段 value/y/val 中读取；
            - 若仍未找到，则在所有 value 中寻找第一个数值字段。
        """

        for key in ("value", "y", "val", "v", "height"):
            if key in meta:
                v = meta[key]
                if isinstance(v, (int, float)):
                    return float(v)
                try:
                    return float(str(v))
                except Exception:  # noqa: BLE001
                    continue

        for v in meta.values():
            if isinstance(v, (int, float)):
                return float(v)
            try:
                return float(str(v))
            except Exception:  # noqa: BLE001
                continue
        return None

    @staticmethod
    def _parse_answer_numeric(answer: str) -> Optional[float]:
        """从答案字符串中解析出一个数值（若存在）。"""

        answer = answer.strip()
        # 去掉常见单位和百分号
        for suf in ("%", "％", "元", "万元", "万", "人", "次"):
            if answer.endswith(suf):
                answer = answer[: -len(suf)]
        try:
            return float(answer)
        except Exception:  # noqa: BLE001
            return None

    def compute_chart_loss_from_samples(
        self,
        samples: List[Dict[str, Any]],
        ignore_index: int = -100,
    ) -> Optional[torch.Tensor]:
        """根据一批 Chart-QA 样本，计算轻量级图表抽取损失。

        思路：
            - 从样本 evidence.chart_elements 中获取图元列表；
            - 对于数值型答案：
                * 从每个元素 meta 中抽取数值，找到与答案数值最接近的元素作为监督；
                * 同时将该元素的数值作为回归目标；
            - 对于非数值答案：
                * 尝试用 label/x/name/category 文本与答案做子串匹配，决定分类目标；
            - 统一使用 ChartHead 计算：
                * 分类损失：选中正确元素；
                * 可选回归损失：预测元素数值（若存在）。
        """

        batch_chart_elements: List[List[Dict[str, Any]]] = []
        class_targets: List[int] = []
        value_targets: List[float] = []
        reg_valid_flags: List[bool] = []

        for ex in samples:
            ev = ex.get("evidence") or {}
            elems = ev.get("chart_elements") or []
            batch_chart_elements.append(elems)

            ans_list = ex.get("answers") or []
            ans = str(ans_list[0]) if ans_list else ""

            class_idx = ignore_index
            reg_value = 0.0
            reg_valid = False

            if elems and ans:
                numeric_ans = self._parse_answer_numeric(ans)
                if numeric_ans is not None:
                    best_idx: Optional[int] = None
                    best_diff: float = float("inf")
                    best_value: Optional[float] = None
                    for idx_elem, elem in enumerate(elems):
                        meta = elem.get("meta") or {}
                        v = self._extract_numeric_from_meta(meta)
                        if v is None:
                            continue
                        diff = abs(v - numeric_ans)
                        if diff < best_diff:
                            best_diff = diff
                            best_idx = idx_elem
                            best_value = v
                    if best_idx is not None and best_value is not None:
                        class_idx = best_idx
                        reg_value = best_value
                        reg_valid = True
                else:
                    # 文本答案：尝试用元素标签/横纵坐标等字段做子串匹配
                    ans_norm = ans.strip()
                    for idx_elem, elem in enumerate(elems):
                        meta = elem.get("meta") or {}
                        label = (
                            str(meta.get("label") or "")
                            or str(meta.get("x") or "")
                            or str(meta.get("name") or "")
                            or str(meta.get("category") or "")
                        ).strip()
                        if not label:
                            continue
                        if label in ans_norm or ans_norm in label:
                            class_idx = idx_elem
                            break

            class_targets.append(class_idx)
            value_targets.append(reg_value)
            reg_valid_flags.append(reg_valid)

        if not any(t != ignore_index for t in class_targets):
            # 没有任何样本提供有效分类监督，则直接跳过该损失
            return None

        chart_hidden = self.encode_chart_elements(batch_chart_elements)  # (B, N, H)
        batch_size, _, hidden_dim = chart_hidden.shape

        # 使用一个全局可训练查询向量作为 pooled_hidden 占位，
        # 后续可替换为基于问题文本/多模态表征的聚合结果。
        if not hasattr(self, "chart_query"):
            self.chart_query = nn.Parameter(torch.zeros(hidden_dim, device=self.device))
            nn.init.normal_(self.chart_query, mean=0.0, std=0.02)

        pooled_hidden = self.chart_query.unsqueeze(0).expand(batch_size, -1)  # (B, H)
        logits, value_pred = self.chart_head(pooled_hidden, chart_hidden)  # (B, N), (B,)

        target_tensor = torch.tensor(class_targets, dtype=torch.long, device=self.device)
        cls_loss = self.chart_head.compute_classification_loss(
            logits,
            target_tensor,
            ignore_index=ignore_index,
        )

        # 对具有数值监督的样本计算回归损失
        reg_mask = torch.tensor(reg_valid_flags, dtype=torch.bool, device=self.device)
        reg_loss: Optional[torch.Tensor] = None
        if reg_mask.any():
            value_target_tensor = torch.tensor(value_targets, dtype=torch.float32, device=self.device)
            reg_loss = self.chart_head.compute_regression_loss(
                value_pred[reg_mask],
                value_target_tensor[reg_mask],
            )

        if reg_loss is not None:
            return cls_loss + reg_loss
        return cls_loss

    # ====================== 任务级接口占位 ======================

    def forward_vqa(
        self,
        texts: List[str],
        images: Optional[List["object"]] = None,
    ) -> torch.Tensor:
        """VQA 任务前向接口（占位实现）。

        当前实现：
            - 仅使用纯文本语言模型 loss；
            - 视觉与桥接层尚未融入实际训练，只作为扩展位存在。
        """

        return self.lm_forward(texts)

    # 后续可以添加 forward_ocr / forward_chart 等方法，
    # 并在其中结合 OCRPointerHead 与 ChartHead 的输出与损失。

    # ====================== 推理阶段辅助接口 ======================

    def predict_answerability(
        self,
        question: str,
        image: Optional["object"] = None,
    ) -> float:
        """基于可答性 Head 预测「当前问题是否值得作答」的概率。

        推理用法：
            - 输入仅包含问题文本与可选图像；
            - 输出一个位于 [0,1] 的分数，表示 answerable=True 的置信度；
            - 该分数会与证据指针置信度结合，用于最终的拒答决策。
        """

        text = f"问题：{question}"
        enc = self.encode_texts([text])
        outputs = self.text_model(
            **enc,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # type: ignore[index]
        pooled_text = hidden_states[:, -1, :]  # (1, H)

        pooled = pooled_text
        if image is not None:
            try:
                vision_feats = self.encode_images([image])  # (1, D_v)
                bridge_out = self.bridge_vision(vision_feats)  # (1, num_q, H)
                pooled_vision = bridge_out.mean(dim=1)  # (1, H)
                if pooled_vision.shape == pooled_text.shape:
                    pooled = pooled_text + pooled_vision
            except Exception:
                pooled = pooled_text

        logits = self.answerability_head(pooled).view(-1)[0]
        prob = torch.sigmoid(logits).item()
        return float(prob)
