# vla-train-eval

JarvisVLA / Minecraft VLA 的训练与评测管线代码（私有存档）。

## 目录结构（保持工作区原始布局，脚本内的绝对路径引用与此对应）

```
evaluation/                 # 评测代码（2026-10 从原位置集中迁移）
├── duplex/                 #   全双工评测: duplex_interact_v4.py, minecraft_eval_protocol.py,
│                           #   vllm_eval_pool.py, 视图构建 prepare_*.py
└── nonduplex/              #   非全双工四任务评测: run_eval_task.py (MODELS 注册表),
                            #   eval_4tasks_group.py, 各模型启动器 run_*.sh / rerun_*.sh

ms-swift/
├── duplex_vla/             # 训练编排与数据构造
│   ├── train_v2_stage.sh   #   单步训练启动器 (支持 TRAIN_ATTN_IMPL 可选 flash-attn)
│   ├── train_v5_stage2_then_duplex.sh       # v5 四阶段序列
│   ├── train_v5_split_action_then_qa.sh     # action/QA 拆分消融序列
│   ├── train_v5_action_only_fa.sh           # flash-attn 速度对比
│   ├── build_duplex_from_stage2_v*.py       # 全双工数据构造
│   ├── split_stage2_v5.py / publish_stage2_v5.py / restore_v5_stage2_noop.py
│   ├── filter_stage2_craft147.py / filter_stage2_trade76.py   # craft 数据清理
│   ├── qa_resample.py      #   Qwen2.5-7B-Instruct 生成 factoid 答案
│   └── expand_ckpt_vocab_v4.py / v5_checkpoint_handoff.py / prepare_v5_split_eval.py
└── mcvla/                  # 训练入口
    ├── run_train.py        #   swift sft 入口 (注册自定义 loss)
    ├── camera_exponential_loss.py / camera_distance_loss.py / action_constrained_loss.py
    └── (数据构建/转换脚本)

JarvisVLA/
└── jarvisvla/              # 底层库
    ├── evaluate/           #   agent_wrapper, 环境构建 (build_env/prepare_episode), 录像
    ├── inference/          #   action_mapping (OneActionTokenizer, 动作 token 151835-151907)
    ├── train/              #   训练侧工具
    └── utils/
```

## 关键约定

- 共享盘 `SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu`（数据/ checkpoint / conda 环境 JarvisVLA、swift）
- 训练: `bash train_v2_stage.sh <model> <data> <output> <steps> [lr] [loss_scale] [loss_type]`，8 卡 ZeRO2，全局 batch 256
- 非全双工评测: `evaluation/nonduplex/run_eval_task.py` 的 MODELS 注册表 + 四任务×3 局 (seed 42/43/44)，vLLM 服务池 4×TP1
- 动作 token: `<|reserved_special_token_178~250|>` = 151835-151907；全双工扩展 `<|silence|>/<|obs_user|>/<|obs_asst|>` = 151908-151910
- 采样: temp 0.9 / top_p 0.99 / top_k -1 / skip_special_tokens False
