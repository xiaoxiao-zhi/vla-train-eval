#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练入口: 注册动作受限 loss 后启动 swift sft (参数与 swift sft 完全一致).

多卡启动由外层 torchrun 包裹 (见 train_mcvla.sh):
    python -m torch.distributed.run --nproc_per_node 8 run_train.py <swift sft 参数...>
"""
import os

from swift.loss import loss_map

from action_constrained_loss import ActionConstrainedLoss
from camera_distance_loss import CameraDistanceLoss
from camera_exponential_loss import CameraExponentialLoss

# 外部注册: trainers/mixin.py 的 loss_map[args.loss_type](args, trainer) 取的是同一 dict
loss_map["action_constrained"] = ActionConstrainedLoss
loss_map["camera_weighted"] = CameraDistanceLoss
loss_map["camera_exponential"] = CameraExponentialLoss

from swift.pipelines import sft_main  # noqa: E402

if __name__ == "__main__":
    if os.environ.get('MCVLA_REPORT_TOKEN_TYPE_ACC', '1') != '0':
        from token_type_accuracy import register_token_type_accuracy

        register_token_type_accuracy()
        print('[metrics] enabled: keyboard_acc, camera_acc, action_acc, language_acc', flush=True)
    sft_main()
