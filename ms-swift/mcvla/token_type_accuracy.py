# -*- coding: utf-8 -*-
"""Token-type accuracy for the external mcvla training entry, without loss changes.

keyboard_acc includes keyboard and mouse-button tokens (suffixes 180..208).
camera_acc includes both axes (209..250), including the zero bins.
action_acc is their union; action delimiters 178/179 are excluded.
language_acc excludes action tokens and tokenizer-declared special tokens.
All categories are defined by gold labels and use full-vocabulary argmax.
The framework's token_acc/seq_acc computation and metric aggregation are reused.
"""
import numpy as np
import torch

from swift.metrics.acc import compute_acc
from swift.trainers import Seq2SeqTrainer
from swift.trainers.trainer_factory import TrainerFactory


class TokenTypeAccuracyTrainer(Seq2SeqTrainer):
    def _token_type_ids(self):
        cached = getattr(self, '_token_type_accuracy_ids', None)
        if cached is not None:
            return cached
        tokenizer = self.template.tokenizer
        ids = {}
        for suffix in range(178, 251):
            token = '<|reserved_special_token_%d|>' % suffix
            token_id = tokenizer.convert_tokens_to_ids(token)
            if (token_id is None or token_id == tokenizer.unk_token_id or token_id < 0
                    or tokenizer.encode(token, add_special_tokens=False) != [token_id]):
                raise ValueError('Missing or non-atomic action token: %s' % token)
            ids[suffix] = token_id
        if len(set(ids.values())) != len(ids):
            raise ValueError('Action tokens must have distinct vocabulary IDs')
        keyboard = np.array([ids[n] for n in range(180, 209)], dtype=np.int64)
        camera = np.array([ids[n] for n in range(209, 251)], dtype=np.int64)
        action = np.concatenate((keyboard, camera))
        excluded = np.array(sorted(set(tokenizer.all_special_ids) | set(ids.values())),
                            dtype=np.int64)
        self._token_type_accuracy_ids = {
            'keyboard': keyboard, 'camera': camera, 'action': action,
            'language_excluded': excluded,
        }
        return self._token_type_accuracy_ids

    @torch.no_grad()
    def _compute_acc(self, outputs, labels, cu_seqlens=None):
        if self.task_type != 'causal_lm' or self.template.sequence_parallel_size != 1:
            return super()._compute_acc(outputs, labels, cu_seqlens=cu_seqlens)

        predictions = outputs.logits.detach().argmax(dim=-1).cpu().numpy()
        targets = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        metrics = compute_acc(
            predictions, targets, acc_strategy=self.args.acc_strategy,
            is_encoder_decoder=self.template.is_encoder_decoder, cu_seqlens=cu_seqlens)
        mode = 'train' if self.model.training else 'eval'
        for name, values in metrics.items():
            self.custom_metrics[mode][name].update(values)
        if not metrics:
            return

        if predictions.ndim >= 2 and not self.template.is_encoder_decoder:
            predictions = predictions[..., :-1]
            targets = targets[..., 1:]
        valid = targets != -100
        ids = self._token_type_ids()
        masks = {
            'keyboard_acc': valid & np.isin(targets, ids['keyboard']),
            'camera_acc': valid & np.isin(targets, ids['camera']),
            'action_acc': valid & np.isin(targets, ids['action']),
            'language_acc': valid & ~np.isin(targets, ids['language_excluded']),
        }
        correct = predictions == targets
        for name, mask in masks.items():
            self.custom_metrics[mode][name].update(correct[mask].astype(np.float32))


def register_token_type_accuracy():
    TrainerFactory.TRAINER_MAPPING['causal_lm'] = 'token_type_accuracy.TokenTypeAccuracyTrainer'
