#!/usr/bin/env python
"""反解 OneActionTokenizer 的 camera 十进制值 -> 实际动作."""
import sys
sys.path.insert(0, '/apdcephfs/private_chengzhixu/JarvisVLA')
from jarvisvla.inference.action_mapping import OneActionTokenizer

tok = OneActionTokenizer(tokenizer_type='qwen2_vl')

# camera 两位进制压缩: value = d10 * 21 + d11 (bases[-2:] = [21, 21])
for v in [198, 219, 220, 240]:
    d10, d11 = divmod(v, 21)  # 逆进制压缩 (近似, 若编码确实是 d10*21+d11)
    print('camera=%d -> bins (d10,d11)=(%d,%d)' % (v, d10, d11))

print()
# 权威验证: 用 null_action 的 encode 看 0 偏移 camera 的编码值
na = tok.null_action
print('null_action:', {k: (v if not hasattr(v, "shape") else v.tolist()) for k, v in na.items()})
null_tokens = tok.null_token()
print('null_token 字符串: %s...' % null_tokens[:80])
ids = tok.tok(null_tokens, add_special_tokens=False)['input_ids'] if hasattr(tok, 'tok') else None
dec = tok.decode(ids)
print('null decode 回:', {k: (v if not hasattr(v, 'shape') else v.tolist()) for k, v in dec[0].items()})
