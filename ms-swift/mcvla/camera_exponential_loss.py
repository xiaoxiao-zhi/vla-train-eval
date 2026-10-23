# -*- coding: utf-8 -*-
"""Opt-in Camera CE reweighting with the existing full-vocabulary Top-5 distance.

Raw weights: 1 + alpha * (1 - exp(-distance / tau)).
Normalize over supervised Camera X/Y positions together in each forward pass.
The default global scope aggregates sum/count across data-parallel ranks, not
across gradient-accumulation steps. Non-Camera CE and the Trainer denominator
are unchanged. Select with --loss_type camera_exponential; omit loss_type to
use the framework's original loss path.
"""
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from camera_distance_loss import CameraDistanceLoss


class CameraExponentialLoss(CameraDistanceLoss):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.alpha = float(os.environ.get('CAMERA_EXP_ALPHA', '1.0'))
        self.tau = float(os.environ.get('CAMERA_EXP_TAU', '4.0'))
        self.normalization = os.environ.get('CAMERA_EXP_NORMALIZATION', 'global')
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('CAMERA_EXP_ALPHA must be finite and >= 0')
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError('CAMERA_EXP_TAU must be finite and > 0')
        if self.normalization not in ('global', 'local'):
            raise ValueError('CAMERA_EXP_NORMALIZATION must be global or local')
        template = getattr(trainer, 'template', None)
        if getattr(template, 'sequence_parallel_size', 1) != 1:
            raise ValueError('camera_exponential supports data parallelism, not sequence parallelism')
        if self.is_megatron:
            raise ValueError('camera_exponential requires the standard swift Trainer')

    def _save_batch_weights(self, labels, camera_weights, loss_scale, details, denominator):
        if not self._log_weights:
            return
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(
            os.environ.get('RANK', getattr(self.args, 'process_index', 0)))
        step = int(self.trainer.state.global_step)
        phase = 'train' if self.trainer.model.training else 'eval'
        key = (phase, step)
        if key != self._step_key:
            self._step_key = key
            self._micro_batch_index = 0
        self._batch_index += 1
        self._micro_batch_index += 1
        output_dir = os.environ.get('CAMERA_WEIGHT_LOG_DIR') or self.trainer.args.output_dir
        log_dir = (Path(output_dir) / 'batch_weights' / ('rank%d_%s' % (rank, self._weight_session))
                   / phase / ('step_%06d' % step))
        log_dir.mkdir(parents=True, exist_ok=True)
        effective = camera_weights if loss_scale is None else camera_weights * loss_scale
        supervised = labels.reshape(-1) != -100
        record = {
            'schema_version': 2,
            'loss_type': 'camera_exponential',
            'rank': rank,
            'session': self._weight_session,
            'phase': phase,
            'batch_index': self._batch_index,
            'global_step_before_update': step,
            'optimizer_step_target': step + 1 if phase == 'train' else None,
            'micro_batch_in_step': self._micro_batch_index,
            'gradient_accumulation_steps': self.trainer.args.gradient_accumulation_steps,
            'shape': list(labels.shape),
            'position_layout': 'shifted_targets_aligned_to_logits_after_trainer_slicing',
            'alpha': self.alpha,
            'tau': self.tau,
            'normalization': self.normalization,
            'normalization_population': 'effective_camera_x_and_y_per_forward',
            'normalization_denominator': denominator.item() if torch.is_tensor(denominator) else denominator,
            'noncamera_weight': 1.0,
            'camera': {},
            'supervised_token_count': supervised.sum().item(),
        }
        for axis, tensors in details.items():
            mask = torch.zeros_like(supervised)
            positions = tensors['positions']
            flat_positions = positions[:, 0] * labels.shape[-1] + positions[:, 1]
            mask[flat_positions] = True
            record['camera'][axis] = {name: tensor.detach().cpu().tolist()
                                      for name, tensor in tensors.items()}
            record['camera'][axis]['loss_scale'] = (None if loss_scale is None
                                                   else loss_scale[mask].detach().cpu().tolist())
            record['camera'][axis]['effective_weights'] = effective[mask].detach().cpu().tolist()
        path = log_dir / ('micro_batch_%03d.json' % self._micro_batch_index)
        temporary = path.with_suffix('.json.tmp')
        with temporary.open('x', encoding='utf-8') as file:
            json.dump(record, file, ensure_ascii=False, allow_nan=False)
        temporary.replace(path)

    def _top5_weights(self, logits, targets, ids, return_details=False):
        with torch.no_grad():
            _, details = super()._top5_weights(logits, targets, ids, return_details=True)
            weights = 1.0 + self.alpha * (-torch.expm1(-details['mean_distance'] / self.tau))
            if return_details:
                details['raw_weights'] = weights
                details['weights'] = weights
                return weights, details
            return weights

    def _camera_weight_mean(self, raw_weights):
        with torch.no_grad():
            totals = torch.stack((raw_weights.double().sum(),
                                  raw_weights.new_tensor(raw_weights.numel(), dtype=torch.float64)))
            if self.normalization == 'global' and dist.is_available() and dist.is_initialized():
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            mean = totals[0] / totals[1].clamp_min(1)
            mean = torch.where(totals[1] > 0, mean, mean.new_ones(()))
            return mean.to(dtype=raw_weights.dtype)

    def __call__(self, outputs, labels, *, num_items_in_batch=None,
                 loss_scale=None, **kwargs):
        logits = outputs.logits.float()
        labels_shift = torch.roll(labels, -1, dims=-1).reshape(-1).to(logits.device)
        logits = logits.reshape(-1, logits.shape[-1])
        loss = F.cross_entropy(logits, labels_shift, ignore_index=-100, reduction='none')
        if loss_scale is not None:
            loss_scale = loss_scale.to(device=logits.device).reshape(-1)
            if loss_scale.shape != loss.shape:
                raise ValueError('loss_scale must already be shifted and aligned by the Trainer')
        if num_items_in_batch is None:
            num_items_in_batch = (labels != -100).sum()

        if self.alpha == 0 and not self._log_weights:
            if loss_scale is not None:
                loss = loss * loss_scale
            return loss.sum() / num_items_in_batch

        if self.axis_ids is None or self._device != logits.device:
            self._build(logits.device)
            print('[camera_exponential] active: alpha=%g tau=%g normalization=%s; '
                  'full-vocabulary Top-5, normalized exponential weights replace linear weights'
                  % (self.alpha, self.tau, self.normalization), flush=True)

        supervised = labels_shift != -100
        if loss_scale is not None:
            supervised = supervised & (loss_scale != 0)
        masks = {}
        raw_by_axis = {}
        batch_details = {}
        for name in ('x', 'y'):
            mask = supervised & torch.isin(labels_shift, self.axis_ids[name])
            masks[name] = mask
            if not mask.any():
                continue
            if self._log_weights:
                weights, detail = self._top5_weights(
                    logits[mask], labels_shift[mask], self.axis_ids[name], return_details=True)
                detail['positions'] = torch.nonzero(mask.reshape(labels.shape), as_tuple=False)
                batch_details['pitch' if name == 'x' else 'yaw'] = detail
            else:
                weights = self._top5_weights(logits[mask], labels_shift[mask], self.axis_ids[name])
            raw_by_axis[name] = weights

        raw_weights = (torch.cat(list(raw_by_axis.values())) if raw_by_axis
                       else loss.new_empty(0))
        weight_mean = self._camera_weight_mean(raw_weights)
        batch_weights = torch.ones_like(loss) if self._log_weights else None
        for name, raw in raw_by_axis.items():
            weights = (raw / weight_mean).detach()
            mask = masks[name]
            loss = loss.masked_scatter(mask, loss[mask] * weights)
            if self._log_weights:
                detail = batch_details['pitch' if name == 'x' else 'yaw']
                detail['weights'] = weights
                detail['normalization_mean'] = weight_mean.expand_as(weights)
                batch_weights = batch_weights.masked_scatter(mask, weights)

        if loss_scale is not None:
            loss = loss * loss_scale
        self._save_batch_weights(labels_shift.reshape(labels.shape), batch_weights,
                                 loss_scale, batch_details, num_items_in_batch)
        return loss.sum() / num_items_in_batch
