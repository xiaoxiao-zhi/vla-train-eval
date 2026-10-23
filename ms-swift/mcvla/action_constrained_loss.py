# -*- coding: utf-8 -*-
"""动作受限交叉熵 loss.

动机: 动作保留字位置的输出空间只可能是 ~80 个 token (动作保留字 + 段标签 + 格式token),
全词表 CE 会迫使模型在 15 万个不可能 token 上浪费容量; 受限 CE 只在候选子集内竞争.

实现: 与 swift/trainers/utils.py 的 per_token_loss_func 相同的 shift/展开约定,
仅对 "label 属于候选集合" 的位置把 logits 切片到候选子集后再算 CE (等价于
其余位置的 logit = -inf), 其余位置 (\n / <|im_end|> / -100) 照常全词表 CE.
"""
import torch
import torch.nn.functional as F

from swift.loss.base import BaseLoss

# 动作保留字范围 (JarvisVLA: 176~250, 含 178/179 段标签)
ACTION_TOKEN_RANGE = range(176, 251)
# 格式 token: 动作段结尾换行 + chat 结束符
FORMAT_TOKENS = ("\n", "<|im_end|>")


class ActionConstrainedLoss(BaseLoss):

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        # 注意: trainer 在构造本对象时 (mixin.py __init__) processing_class 尚未赋值,
        # 因此候选 token 集延迟到第一次 __call__ 时再从 tokenizer 构建
        self.cand_ids = None
        self._cand_device = None

    def _build_cand_ids(self):
        import torch
        tokenizer = self.trainer.tokenizer  # 训练开始后可用

        cand = []
        for n in ACTION_TOKEN_RANGE:
            tid = tokenizer.convert_tokens_to_ids("<|reserved_special_token_%d|>" % n)
            if tid is not None and tid >= 0:
                cand.append(tid)
        for tok in FORMAT_TOKENS:
            cand += tokenizer.encode(tok, add_special_tokens=False)
        cand = sorted(set(cand))
        assert len(cand) > 70, "candidate set too small: %d" % len(cand)

        self.cand_ids = torch.tensor(cand, dtype=torch.long)
        print("[action_constrained] %d candidate token ids (head=%s)"
              % (len(cand), cand[:5]))

    def _cand_on(self, device):
        if self.cand_ids is None:
            self._build_cand_ids()
        if self._cand_device != device:
            self.cand_ids = self.cand_ids.to(device)
            self._cand_device = device
        return self.cand_ids

    def __call__(self, outputs, labels, *, num_items_in_batch=None,
                 loss_scale=None, **kwargs):
        logits = outputs.logits.float()                    # (B, L, V)
        # 与 per_token_loss_func 一致: labels 左移一位对齐 next-token 预测
        labels_shift = torch.roll(labels, -1, dims=-1).view(-1).to(logits.device)
        logits = logits.view(-1, logits.shape[-1])         # (N, V)

        cand_ids = self._cand_on(logits.device)
        is_cand = torch.isin(labels_shift, cand_ids)       # -100 不会命中

        # 1) 全词表 CE (非动作位置: \n / <|im_end|> 等)
        loss = F.cross_entropy(logits, labels_shift,
                               ignore_index=-100, reduction="none")
        # 2) 受限 CE (动作位置): 只在候选子集内 softmax
        if is_cand.any():
            sub_logits = logits[is_cand][:, cand_ids]      # (Nc, C)
            sub_target = torch.searchsorted(cand_ids, labels_shift[is_cand])
            sub_loss = F.cross_entropy(sub_logits, sub_target, reduction="none")
            loss = loss.masked_scatter(is_cand, sub_loss)

        if loss_scale is not None:
            loss = loss * loss_scale
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        return loss.sum() / num_items_in_batch
