# -*- coding: utf-8 -*-
"""CPU 验证乘法加权 CE：全部档位、非法预测、mask、梯度及真实 tokenizer。"""
import os
import re
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from camera_distance_loss import CameraDistanceLoss


class FakeTok:
    def convert_tokens_to_ids(self, token):
        match = re.fullmatch(r'<\|reserved_special_token_(\d+)\|>', token)
        if match and 209 <= int(match.group(1)) <= 250:
            return int(match.group(1)) - 199
        return None

    def encode(self, token, add_special_tokens=False):
        return [self.convert_tokens_to_ids(token)]


def make_loss(tokenizer=None):
    return CameraDistanceLoss(SimpleNamespace(),
                              SimpleNamespace(tokenizer=tokenizer or FakeTok()))


def reference_loss(logits, labels, axis_ids, scale=None, denominator=None):
    vocab_size = logits.shape[-1]
    targets = torch.roll(labels, -1, dims=-1).flatten()
    flat = logits.float().reshape(-1, vocab_size)
    ce = F.cross_entropy(flat, targets, ignore_index=-100, reduction='none')
    distributions = F.softmax(flat.detach().double(), dim=-1)
    weights = []
    for target, distribution in zip(targets.tolist(), distributions):
        weight = 1.0
        for ids in axis_ids.values():
            bins = ids.tolist()
            if target in bins:
                probabilities, predictions = distribution.topk(5)
                probabilities = (probabilities / probabilities.sum()).tolist()
                distance = sum(probability * (
                    abs(bins.index(predicted) - bins.index(target)) if predicted in bins else 20
                ) for probability, predicted in zip(probabilities, predictions.tolist()))
                weight = 0.5 + max(distance - 1, 0) * 1.5 / 19.0
                break
        weights.append(weight)
    per_token = ce * ce.new_tensor(weights)
    if scale is not None:
        per_token = per_token * scale
    if denominator is None:
        denominator = (labels != -100).sum()
    return per_token.sum() / denominator


class CameraDistanceLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.loss_fn = make_loss()
        self.loss_fn._build(torch.device('cpu'))

    def test_all_axis_weights(self):
        for axis in ('x', 'y'):
            matrix = self.loss_fn.axis_w[axis]
            for gold in range(21):
                for predicted in range(21):
                    distance = abs(predicted - gold)
                    expected = 0.5 + max(distance - 1, 0) * 1.5 / 19.0
                    self.assertAlmostEqual(matrix[gold, predicted].item(), expected, places=6)
            torch.testing.assert_close(matrix, matrix.T)

    def test_all_predictions_multiply_ce(self):
        for axis in ('x', 'y'):
            ids = self.loss_fn.axis_ids[axis]
            gold = ids.repeat_interleave(21)
            predicted = ids.repeat(21)
            logits = torch.full((441, 2, 64), -50.0)
            logits[torch.arange(441), 0, predicted] = 50.0
            labels = torch.full((441, 2), -100, dtype=torch.long)
            labels[:, 1] = gold
            scale = torch.zeros(441, 2)
            scale[:, 0] = 1.0
            distance = (torch.arange(21).repeat_interleave(21)
                        - torch.arange(21).repeat(21)).abs()
            ce = F.cross_entropy(logits[:, 0], gold, reduction='none')
            for error in range(21):
                selected = distance == error
                masked_scale = (scale * selected[:, None]).flatten()
                actual = self.loss_fn(SimpleNamespace(logits=logits), labels,
                                      loss_scale=masked_scale,
                                      num_items_in_batch=selected.sum())
                weight = 0.5 + max(error - 1, 0) * 1.5 / 19.0
                torch.testing.assert_close(actual, (ce[selected] * weight).mean())

    def test_alignment_mask_and_gradients(self):
        labels = torch.tensor([
            [-100, 1, 3, 10, 51, 2, 4, -100],
            [-100, 1, 30, 31, 2, 4, -100, -100],
        ])
        logits = torch.randn(2, 8, 64, requires_grad=True)
        scale = torch.tensor([
            1.0, 0.25, 2.0, 0.0, 1.5, 1.0, 0.0, 0.0,
            0.5, 2.0, 0.75, 1.0, 1.0, 0.0, 0.0, 0.0,
        ])
        actual = self.loss_fn(SimpleNamespace(logits=logits), labels,
                              loss_scale=scale, num_items_in_batch=37)
        expected = reference_loss(logits, labels, self.loss_fn.axis_ids, scale, 37)
        torch.testing.assert_close(actual, expected)
        actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertTrue(torch.isfinite(actual_gradient).all())
        self.assertGreater(actual_gradient.abs().sum().item(), 0)
        flat_gradient = actual_gradient.reshape(-1, 64)
        self.assertEqual(flat_gradient[scale == 0].abs().sum().item(), 0)
        unscaled = self.loss_fn(SimpleNamespace(logits=logits), labels)
        torch.testing.assert_close(unscaled,
                                   reference_loss(logits, labels, self.loss_fn.axis_ids))

    def test_noncamera_tokens_unchanged(self):
        labels = torch.tensor([[-100, 1, 3, 2, 4, -100]])
        logits = torch.randn(1, 6, 64, requires_grad=True)
        actual = self.loss_fn(SimpleNamespace(logits=logits), labels)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, 64),
                             labels[:, 1:].reshape(-1), ignore_index=-100, reduction='sum')
        torch.testing.assert_close(actual, ce / (labels != -100).sum())

    def test_invalid_and_cross_axis_prediction(self):
        for predicted in (3, 31):
            logits = torch.full((1, 2, 64), -50.0)
            logits[0, 0, predicted] = 50.0
            labels = torch.tensor([[-100, 10]])
            ce = F.cross_entropy(logits[:, 0], labels[:, 1])
            actual = self.loss_fn(SimpleNamespace(logits=logits), labels)
            torch.testing.assert_close(actual, ce * 2.0)

    def test_top5_mean_distance_before_weight(self):
        logits = torch.full((1, 2, 64), -50.0, requires_grad=True)
        top_ids = [20, 21, 25, 31, 3]
        probabilities = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05])
        with torch.no_grad():
            logits[0, 0, top_ids] = probabilities.log()
            logits[0, 0, 30] = torch.tensor(0.04).log()
        labels = torch.tensor([[-100, 20]])
        mean_distance = 0.4 * 0 + 0.3 * 1 + 0.15 * 5 + 0.1 * 20 + 0.05 * 20
        expected_weight = 0.5 + (mean_distance - 1) * 1.5 / 19
        weights = self.loss_fn._top5_weights(logits[:, 0], labels[:, 1],
                                            self.loss_fn.axis_ids['x'])
        torch.testing.assert_close(weights, torch.tensor([expected_weight]))
        self.assertFalse(weights.requires_grad)
        ce = F.cross_entropy(logits[:, 0], labels[:, 1])
        actual = self.loss_fn(SimpleNamespace(logits=logits), labels)
        torch.testing.assert_close(actual, ce * expected_weight)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    @unittest.skipUnless(os.environ.get('CAMERA_TEST_CHECKPOINT'), '未指定真实checkpoint')
    def test_real_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.environ['CAMERA_TEST_CHECKPOINT'],
                                                  trust_remote_code=True, local_files_only=True)
        loss_fn = make_loss(tokenizer)
        loss_fn._build(torch.device('cpu'))
        gold = loss_fn.axis_ids['x'][1].item()
        predicted = loss_fn.axis_ids['x'][11].item()
        logits = torch.zeros(1, 2, len(tokenizer), requires_grad=True)
        with torch.no_grad():
            logits[0, 0, predicted] = 3.0
        labels = torch.tensor([[-100, gold]])
        actual = loss_fn(SimpleNamespace(logits=logits), labels)
        expected = reference_loss(logits, labels, loss_fn.axis_ids)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
