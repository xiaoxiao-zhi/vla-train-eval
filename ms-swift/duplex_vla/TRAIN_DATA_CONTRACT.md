# 全双工数据训练消费契约（B 方案：id 级自定义 Dataset）

> 数据: `dataset/duplex_vla/duplex_turns_train.jsonl` (每行一个时间步样本)
> 词表: 基座官方 `JarvisVLA-Qwen2-VL-7B` + 5 个控制 token (见 `duplex_tokens.json`)
> 本文档是训练侧实现的数据契约, 训练代码 (Dataset/Collator/Trainer) 按此实现.

## 1. 前置: 扩词表副本 (一次性)

```
HF_MODEL/JarvisVLA-Qwen2-VL-7B-duplex/
  = 官方模型全部文件
  + tokenizer 注册 5 个控制 token (id 151908~151912, 与 duplex_tokens.json 一致)
  + resize_token_embeddings(151913) 后的权重 (5 行新 embedding 随机初始化, 参与训练)
```

## 2. 控制 token (duplex_tokens.json, 已固定)

| token | id |
|---|---|
| `<\|duplex_user\|>` | 151908 |
| `<\|duplex_obs\|>`  | 151909 |
| `<\|duplex_act\|>`  | 151910 |
| `<\|duplex_txt\|>`  | 151911 |
| `<\|silence\|>`     | 151912 |

空槽位在数据中的表示: `[151912]` (单元素列表).

## 3. 渲染规则 (jsonl 样本 → token 序列, 纯查表拼接, 不经过 tokenizer 文本编码)

对样本按以下顺序拼 id (chat template 的 system/im_start/im_end 外壳由模板层统一加):

```
for c in context (按 i 升序):
    [duplex_user] + c.user_ids
    if c.frame is not None:
        [duplex_obs] + [image_pad] * N(c.frame) + [duplex_act] + c.action_ids
    [duplex_txt] + c.txt_ids
当前轮输入:
    [duplex_user] + current.user_ids + [duplex_obs] + [image_pad] * N(current.frame)
目标段 (唯一进 labels 的部分):
    [duplex_act] + target.action_ids + [duplex_txt] + target.txt_ids + [im_end]
```

- `image_pad` id = tokenizer 的 `<|image_pad|>`
- `N(frame)`: 该帧展开的 image token 数, **运行时**由 Qwen2VL image processor 计算
  (`grid_thw` 三元组的乘积 / (merge_size^2 * temporal_patch_size 相关)), 不预存于数据
- 帧文件从 `images` 列表按序读取 (与序列中出现的顺序一致), 相对 `mcvla_sft/frames/`

## 4. loss mask

```
context + 当前轮输入部分:  labels = -100
目标段 (act+txt+im_end):   labels = 原始 id
```

即每条样本只对当前轮的动作+文本输出算 loss (历史全部只作条件).

## 5. 训练接口 (自定义 Dataset/Collator)

```
Dataset.__getitem__(i) ->
    input_ids:   List[int]           # 渲染后的完整序列 (含图像展开)
    labels:      List[int]           # 同长, 目标段外为 -100
    pixel_values: Tensor              # images 列表过 image processor
    image_grid_thw: Tensor            # 对应 grid 信息
Collator: batch 内按最长 pad (labels pad -100, attention_mask 相应构造)
Trainer: transformers Trainer + deepspeed zero2 (stage2 超参可平移: lr 1e-5,
         grad_accum, cosine, warmup 等)
```

## 6. 严格约束 (不可违反)

1. **txt_ids 的 id 序列不可重新分词**: 一切拼接在 id 空间完成, 禁止 decode→encode 往返
   (txt 段是答案 id 的硬切片, 常落在词中间, 文本往返不保真)
2. 等长双通道: QA 期间每轮 len(txt_ids)==len(action_ids) (收尾轮允许短) — 数据已保证,
   训练侧只需原样消费
3. 控制 token id 以 `duplex_tokens.json` 为唯一事实源 (勿在训练代码中重新计算)
4. 本数据集只能配官方 JarvisVLA-Qwen2-VL-7B 词表基座 (len=151908);
   换基座须重分配控制 token id 并重建数据
