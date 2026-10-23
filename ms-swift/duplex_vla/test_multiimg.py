"""实测: swift qwen2_vl 模板对多图消息的渲染 (多 <image> 标签 + images 列表).

分别测 3 种写法, 看哪种是 swift 支持的标准格式:
  A. 一条 user 消息里多个 <image> 标签
  B. 多条消息各带一个 <image> (已知支持, hist1 格式)
  C. A + images 平铺
"""
from swift.template import get_template
from transformers import AutoProcessor

BASE = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
        'chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B-silence')
processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
template = get_template(processor, template_type='qwen2_vl')
template.set_mode('train')

F = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
     'chengzhixu/dataset/mcvla_video_sft/frames/brandon-f153ac423f61-20211116-135137-1837-1907_f000000.jpg')

def render(msgs, images):
    out = template.encode({'messages': msgs, 'images': images})
    txt = processor.tokenizer.decode(out['input_ids'], skip_special_tokens=False)
    txt = txt.replace('<|image_pad|>', '[IMG]')
    # 图片 token 折叠计数
    import re
    txt = re.sub(r'(\[IMG\])+', lambda m: '[IMG*%d]' % (len(m.group(0)) // 5), txt)
    n_label = sum(1 for x in out.get('labels', []) if x != -100)
    return txt, len(out['input_ids']), n_label

# 写法 A: 单条消息 3 个 <image> 标签
msgsA = [
    {'role': 'user', 'content': 'Kill the zombie.\nobservation: \n<image>\n<image>\n<image>'},
    {'role': 'assistant', 'content': '<|reserved_special_token_178|><|reserved_special_token_219|><|reserved_special_token_240|><|reserved_special_token_179|><|silence|>'},
]
txtA, nA, lA = render(msgsA, [F, F, F])  # 同一图重复用 (只验证格式)
print('=== A: 单消息 3 个 <image> 标签 ===')
print(repr(txtA[:400]))
print('input_ids=%d, labels有效=%d\n' % (nA, lA))

# 写法 B: 3 条消息各 1 图 (旧 hist1 风格, 对照)
msgsB = [
    {'role': 'user', 'content': 'Kill the zombie.\nobservation: \n<image>'},
    {'role': 'assistant', 'content': 'ACT<|silence|>'},
    {'role': 'user', 'content': 'observation: \n<image>'},
    {'role': 'assistant', 'content': 'ACT<|silence|>'},
    {'role': 'user', 'content': 'observation: \n<image>'},
    {'role': 'assistant', 'content': 'ACT<|silence|>'},
]
txtB, nB, lB = render(msgsB, [F, F, F])
print('=== B: 3 轮各 1 图 (对照) ===')
print('input_ids=%d, labels有效=%d' % (nB, lB))
