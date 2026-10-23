#!/usr/bin/env python
import json
from pathlib import Path

BASE = Path('/apdcephfs/private_chengzhixu/JarvisVLA/outputs/qa/trivia_validation200')
LATEST = Path('/apdcephfs/private_chengzhixu/JarvisVLA/outputs/qa/trivia_validation200_latest')

BASELINE_CORRECT = {
    0, 1, 2, 6, 7, 8, 9, 12, 13, 14, 22, 24, 25, 28, 29, 30, 33, 34, 35,
    37, 38, 40, 45, 46, 48, 49,
    51, 52, 53, 54, 56, 62, 63, 66, 67, 69, 70, 71, 76, 77, 78, 79, 81,
    82, 83, 84, 85, 86, 89, 90, 92, 93, 95, 96, 98,
    102, 105, 106, 109, 110, 111, 113, 115, 117, 118, 120, 121, 123, 124,
    125, 126, 129, 130, 131, 132, 133, 135, 138, 139, 141, 142, 143, 145,
    147, 148,
    151, 154, 155, 157, 158, 159, 166, 167, 168, 169, 170, 171, 173, 175,
    177, 178, 180, 184, 185, 186, 187, 188, 190, 195, 196, 197, 198,
}
LATEST_CORRECT = {
    0, 1, 2, 3, 5, 6, 7, 9, 12, 13, 14, 16, 22, 24, 27, 28, 29, 30, 33,
    34, 35, 37, 38, 40, 41, 45, 46, 48, 49,
    51, 54, 55, 58, 59, 60, 61, 62, 63, 66, 67, 69, 70, 71, 76, 78, 79,
    81, 82, 83, 84, 85, 86, 89, 90, 91, 93, 95, 96, 97, 98,
    100, 102, 104, 105, 106, 108, 109, 110, 113, 115, 117, 118, 120, 121,
    122, 123, 124, 125, 126, 127, 129, 130, 131, 132, 133, 135, 137, 139,
    141, 145, 147, 149,
    150, 151, 152, 155, 157, 159, 165, 166, 167, 168, 169, 170, 171, 173,
    175, 177, 178, 180, 181, 182, 184, 187, 188, 189, 190, 191, 193, 195,
    196, 197, 198,
}
EXCLUDED = {
    10: '题目声称某现象仅在日全食可见，但参考答案与日冕这一可接受替代不构成唯一判分；两模型均从比较分母排除。',
    36: '题面声称四个衰变系列都终止于同一元素，忽略镎系终点差异，参考答案不适用于全部系列；两模型均排除。',
    156: '题面缺失必要年份，不能唯一确定；两模型均排除。',
}
NOTES = {
    4: '最新模型把藤壶也归入Malacostraca，分类核心错误，提到crustaceans不等于正确。',
    8: '最新模型先把目标指为Candide本人，随后提Pangloss，核心指称矛盾，判错。',
    18: '普通模型只叙述小说，未给达尔文雀的连接；最新模型未在已保存输出中完成连接结论，不能代模型补答案。',
    34: '两模型都给出正确生卒年范围，最新另给1735这一生前年份，满足问题核心要求。',
    41: '最新回答明确说明贝壳制成珠串，符合beads or shells；字符串包含漏判，不以其他补充细节扣核心分。',
    44: '最新把Travelocity误答为Expedia；Roaming Gnome是重复题目和宽泛别名，不算正确。',
    47: '最新把差异答成含糖浓度和人工甜味剂，又声称两者都用相同糖，未正确给出蔗糖相对高果糖玉米糖浆的区别。',
    53: '最新只谈早教项目和alphabet词源，没有回答alphabetical组织方式。',
    57: '最新给periodontitis，不满足题目八字母要求；pyorrhea才是题目所求。',
    68: '最新虽提到组合，但直接颠倒本名和艺名关系，核心关系错误。',
    77: '最新否定题目中的尺度对应，只列最高类别数字和错误风速，未正确建立hurricane对应12级。',
    78: '普通回答The Robber仅漏复数s，明确指该作品，接受轻微变体。',
    98: 'Handbag比gold的Bag更具体，语义正确。',
    100: '最新答Bo Duke和Luke Duke，插入姓氏不改变正确的两人指称。',
    106: 'Mandible即下颌骨，接受同义词。',
    109: '题目未限定磨刀棒，Stone在该语境指磨刀石，也是合理正确工具。',
    117: '两模型均明确登顶珠穆朗玛峰，Mount插词和climb词形变化不算错误。',
    132: 'John在题面语境明确指Saint John，接受省略称号。',
    138: '最新并列不正确释义，又拒绝明确判断，不视为一致的接吻结论。',
    147: 'Lily-of-the-Valley与Lily of the Valley同指铃兰；连字符不改变语义。',
    151: '两模型均明确Apollo 11，最新角色描述错误属于额外解释，不扣所问任务核心分。',
    160: '最新误认Master，重复Creator of the Daleks不能当命中。',
    167: 'mit是mitt轻微拼写误差且明确指棒球手套，接受；英国英语说法另属错误补充。',
    172: '题面金牌赛事表述有误，但两模型举重均不是所问运动员的铅球项目，核心仍错。',
    185: '最新将Tandy Research等同Microsoft，核心公司身份不一致，不因提到Microsoft判对。',
    199: '最新答塑形束腰，未答gold所要求的类型；重复Florentine girdle不能计作正确。',
}


def load(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    indexed = {row['idx']: row for row in records}
    if len(records) != 200 or set(indexed) != set(range(200)):
        raise ValueError('Results must contain all 200 unique indices')
    return indexed


def main():
    a, b = load(BASE / 'results.jsonl'), load(LATEST / 'results.jsonl')
    for idx in a:
        if any(a[idx][key] != b[idx][key] for key in ('question', 'gold', 'aliases', 'messages', 'source_id')):
            raise ValueError('Sample mismatch at idx=%d' % idx)
    if BASELINE_CORRECT & EXCLUDED.keys() or LATEST_CORRECT & EXCLUDED.keys():
        raise ValueError('Excluded items cannot be marked correct')
    audit_a = json.loads((BASE / 'summary.json').read_text())
    audit_b = json.loads((LATEST / 'summary.json').read_text())
    assert audit_a['samples_sha256'] == audit_b['samples_sha256']
    rows = []
    for idx in range(200):
        verdict_a = None if idx in EXCLUDED else int(idx in BASELINE_CORRECT)
        verdict_b = None if idx in EXCLUDED else int(idx in LATEST_CORRECT)
        rows.append({'idx': idx, 'source_id': a[idx]['source_id'],
                     'baseline_core_correct': verdict_a, 'latest_core_correct': verdict_b,
                     'note': EXCLUDED.get(idx, NOTES.get(idx, ''))})
    valid_n = 200 - len(EXCLUDED)
    n_a, n_b = len(BASELINE_CORRECT), len(LATEST_CORRECT)
    summary = {
        'review_type': 'AI-assisted per-item semantic review; not independently human-annotated ground truth',
        'policy': 'Judge the requested core answer, accept clear synonyms and minor spelling variants; do not deduct separate explanatory errors. Incorrect or contradictory core conclusions, echoing question aliases, or unfinished inference count as wrong.',
        'n_total': 200, 'n_scored': valid_n, 'excluded_indices': list(EXCLUDED),
        'baseline_model': audit_a['model'], 'latest_model': audit_b['model'],
        'samples_sha256': audit_a['samples_sha256'],
        'baseline_correct_n': n_a, 'baseline_accuracy': n_a / valid_n,
        'latest_correct_n': n_b, 'latest_accuracy': n_b / valid_n,
        'latest_minus_baseline_percentage_points': 100 * (n_b - n_a) / valid_n,
        'paired': {'both_correct_n': len(BASELINE_CORRECT & LATEST_CORRECT),
                   'baseline_only_correct_n': len(BASELINE_CORRECT - LATEST_CORRECT),
                   'latest_only_correct_n': len(LATEST_CORRECT - BASELINE_CORRECT),
                   'both_wrong_n': valid_n - len(BASELINE_CORRECT | LATEST_CORRECT)},
        'limitation': 'Review involves semantic and ambiguous-case judgments. This small held-out sample does not establish population-level superiority or whole-response factual correctness.',
        'per_item_verdicts': rows,
    }
    path = LATEST / 'semantic_review.json'
    with open(path, 'x') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != 'per_item_verdicts'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
