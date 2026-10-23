'''
Author: Muyao 2350076251@qq.com
Date: 2025-03-05 10:56:23
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-05 11:01:23
'''


def load_visual_model(checkpoint_path ="",**kwargs):
    if not checkpoint_path:
        raise AssertionError("checkpoint_path is required")
    
    checkpoint_path = checkpoint_path.lower().replace('-','_')
    
    if "qwen2_vl" in checkpoint_path:
        LLM_backbone = "qwen2_vl"
        VLM_backbone = "qwen2_vl"
        return LLM_backbone,VLM_backbone
    else:
        # 默认 qwen2_vl: 当前所有模型 (官方/微调) 均为 Qwen2-VL 架构,
        # 路径名匹配仅作显式标识用, 不再因目录名不含 qwen2_vl 而报错
        return "qwen2_vl", "qwen2_vl"
        