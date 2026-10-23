# -*- coding: utf-8 -*-
"""camera Top5 平均档位偏差乘法加权交叉熵。

从全词表取 Top5，将这5个候选的概率重新归一化为 q。
同轴候选距离为 abs(predicted_bin - gold_bin)，其他候选距离为20。
先求 mean_distance = sum(q * distance)，再计算
weight = 0.5 + max(mean_distance - 1, 0) * 1.5 / 19。
仅有效 camera 目标位置的 CE 乘 weight，其他目标位置权重为1。
Top5、概率和权重均不参与求导；保留原始全词表 CE 和 loss_scale。
用法: --loss_type camera_weighted，可叠加 --loss_scale last_round。
"""

import atexit
import json
import os
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F

from swift.loss.base import BaseLoss

CAM_X_SUFFIX = range(209, 230)   # 垂直轴 pitch, 21 档
CAM_Y_SUFFIX = range(230, 251)   # 水平轴 yaw, 21 档
W_MIN, W_MAX = 0.5, 2.0          # 差1档 -> 0.5, 差20档 -> 2.0


class CameraDistanceLoss(BaseLoss):
    """仅对 camera 位置按预测档位差乘法加权。"""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.axis_ids = None
        self.axis_w = None
        self._device = None
        self._log_weights = os.environ.get('CAMERA_LOG_BATCH_WEIGHTS', '1') != '0'
        self._weight_log = None
        self._weight_session = uuid.uuid4().hex[:12]
        self._batch_index = 0
        self._step_key = None
        self._micro_batch_index = 0
        if self._log_weights:
            atexit.register(self._close_weight_log)

    def _close_weight_log(self):
        if self._weight_log is not None:
            self._weight_log.close()
            self._weight_log = None

    def _save_batch_weights(self, labels, camera_weights, loss_scale, details, denominator):
        if not self._log_weights:
            return
        rank = int(os.environ.get('RANK', getattr(self.args, 'process_index', 0)))
        if self._weight_log is None:
            output_dir = os.environ.get('CAMERA_WEIGHT_LOG_DIR') or self.trainer.args.output_dir
            log_dir = Path(output_dir) / 'batch_weights'
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / ('rank%d_%s.jsonl' % (rank, self._weight_session))
            self._weight_log = path.open('x', encoding='utf-8')
            print('[camera_distance] batch weights log: %s' % path, flush=True)
        step = int(self.trainer.state.global_step)
        phase = 'train' if self.trainer.model.training else 'eval'
        key = (phase, step)
        if key != self._step_key:
            self._weight_log.flush()
            self._step_key = key
            self._micro_batch_index = 0
        self._batch_index += 1
        self._micro_batch_index += 1
        shape = list(labels.shape)
        valid = labels != -100
        effective_weights = camera_weights if loss_scale is None else camera_weights * loss_scale
        record = {
            'schema_version': 1,
            'rank': rank,
            'session': self._weight_session,
            'batch_index': self._batch_index,
            'phase': phase,
            'global_step_before_update': step,
            'optimizer_step_target': step + 1 if phase == 'train' else None,
            'micro_batch_in_step': self._micro_batch_index,
            'gradient_accumulation_steps': self.trainer.args.gradient_accumulation_steps,
            'shape': shape,
            'position_layout': 'shifted_targets_aligned_to_logits_after_trainer_slicing',
            'target_ids': labels.detach().cpu().tolist(),
            'camera_weights': camera_weights.reshape(shape).detach().cpu().tolist(),
            'loss_scale': None if loss_scale is None else loss_scale.reshape(shape).detach().cpu().tolist(),
            'effective_weights': effective_weights.reshape(shape).masked_fill(~valid, 0).detach().cpu().tolist(),
            'normalization_denominator': denominator.item() if torch.is_tensor(denominator) else denominator,
            'camera': {},
        }
        for axis, tensors in details.items():
            record['camera'][axis] = {name: tensor.detach().cpu().tolist()
                                      for name, tensor in tensors.items()}
        self._weight_log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self._weight_log.flush()

    def _build(self, device):
        tokenizer = self.trainer.tokenizer  # 训练开始后可用
        ids = {}
        for name, rng in (('x', CAM_X_SUFFIX), ('y', CAM_Y_SUFFIX)):
            row = []
            for n in rng:
                tid = tokenizer.convert_tokens_to_ids(
                    '<|reserved_special_token_%d|>' % n)
                assert tid is not None and tid >= 0, 'camera token 缺失: %d' % n
                assert tokenizer.encode('<|reserved_special_token_%d|>' % n,
                                        add_special_tokens=False) == [tid], 'camera token 非单token: %d' % n
                row.append(tid)
            assert len(set(row)) == 21, 'camera token ID 重复: %s' % name
            assert row == sorted(row), 'camera token ID 未按档位递增: %s' % name
            ids[name] = torch.tensor(row, dtype=torch.long, device=device)
        d = (torch.arange(21).unsqueeze(0) - torch.arange(21).unsqueeze(1)).abs()
        w = W_MIN + (d - 1).clamp_min(0).float() * (W_MAX - W_MIN) / 19.0
        self.axis_ids = ids
        self.axis_w = {k: w.to(device) for k in ids}
        self._device = device
        print('[camera_distance] x轴起始id=%d y轴起始id=%d | CE乘权重: w(0)=%.4f w(1)=%.4f w(20)=%.1f'
              % (ids['x'][0].item(), ids['y'][0].item(),
                 w[0, 0].item(), w[0, 1].item(), w[0, 20].item()))

    def _top5_weights(self, logits, targets, ids, return_details=False):
        with torch.no_grad():
            top_logits, top_ids = logits.topk(5, dim=-1)
            probabilities = F.softmax(top_logits.float(), dim=-1)
            distances = torch.full_like(probabilities, 20.0)
            valid = torch.isin(top_ids, ids)
            gold_bins = torch.searchsorted(ids, targets)
            gold_bins = gold_bins[:, None].expand_as(top_ids)
            predicted_bins = torch.searchsorted(ids, top_ids[valid])
            distances[valid] = (predicted_bins - gold_bins[valid]).abs().float()
            mean_distance = (probabilities * distances).sum(-1).clamp(0.0, 20.0)
            weights = (W_MIN + (mean_distance - 1.0).clamp_min(0.0)
                       * (W_MAX - W_MIN) / 19.0).clamp(W_MIN, W_MAX)
            if return_details:
                return weights, {
                    'target_ids': targets,
                    'gold_bins': torch.searchsorted(ids, targets),
                    'top5_ids': top_ids,
                    'top5_probabilities_normalized': probabilities,
                    'top5_distances': distances,
                    'mean_distance': mean_distance,
                    'weights': weights,
                }
            return weights

    def __call__(self, outputs, labels, *, num_items_in_batch=None,
                 loss_scale=None, **kwargs):
        logits = outputs.logits.float()                    # (B, L, V)
        # 与 per_token_loss_func 相同: labels 左移一位对齐 next-token 预测
        labels_shift = torch.roll(labels, -1, dims=-1).view(-1).to(logits.device)
        logits = logits.view(-1, logits.shape[-1])         # (N, V)
        if self.axis_ids is None or self._device != logits.device:
            self._build(logits.device)

        # 1) 标准全词表 CE (与 swift 默认完全一致)
        loss = F.cross_entropy(logits, labels_shift,
                               ignore_index=-100, reduction='none')

        batch_weights = torch.ones_like(loss) if self._log_weights else None
        batch_details = {}
        for name in ('x', 'y'):
            ids = self.axis_ids[name]
            is_cam = torch.isin(labels_shift, ids)
            if not is_cam.any():
                continue
            if self._log_weights:
                weights, detail = self._top5_weights(logits[is_cam], labels_shift[is_cam],
                                                    ids, return_details=True)
                detail['positions'] = torch.nonzero(is_cam.reshape(labels.shape), as_tuple=False)
                batch_details['pitch' if name == 'x' else 'yaw'] = detail
                batch_weights = batch_weights.masked_scatter(is_cam, weights)
            else:
                weights = self._top5_weights(logits[is_cam], labels_shift[is_cam], ids)
            loss = loss.masked_scatter(is_cam, loss[is_cam] * weights)

        if loss_scale is not None:
            loss = loss * loss_scale
        if num_items_in_batch is None:
            # 与 swift 默认路径严格一致 (seq2seq_trainer.py: labels != -100 全位置计数)
            num_items_in_batch = (labels != -100).sum()
        self._save_batch_weights(labels_shift.reshape(labels.shape), batch_weights,
                                 loss_scale, batch_details, num_items_in_batch)
        return loss.sum() / num_items_in_batch
