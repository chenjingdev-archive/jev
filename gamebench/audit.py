"""억지스러운 문항 점검 (README "억지스러운 문항 점검" 절).

1. 문항 묶음을 빼고 다시 채점해 순위가 바뀌는지 본다. 점수 계산은 report.py와 같다
   (문항별 우연 보정 점수 → 게임 평균 → 게임 평균의 평균).
2. ② 변형에서 틀린 답만 놓고 원래 규칙의 답을 고른 비율을 본다. 틀린 답이 무작위라면
   기대값은 문항마다 1/(n-1)의 평균이다.

  .venv/bin/python gamebench/audit.py --systems jev-1.13.0 kev semif ...
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score import score_system  # noqa: E402

# 단어나 관습의 뜻을 정반대로 바꾼 변형 (보 > 가위, 업 = 더 작다, A가 가장 낮다)
FLIP_MEANING = {'rps', 'hanabbaegi', 'rps_pattern', 'updown', 'hilo'}
# 이기는 조건과 지는 조건을 맞바꾼 변형
FLIP_WIN = {'mukjjippa', 'nim', 'br31', 'subtraction'}
# 놀이라기보다 퍼즐에 가까운 게임
PUZZLE = {'payoff', 'rps_pattern', 'subtraction'}

variant = lambda r: r['stage'].startswith('2')
CUTS = {
    'all': lambda r: True,
    'no_variant': lambda r: not variant(r),
    'no_flip_meaning': lambda r: not (variant(r) and r['game'] in FLIP_MEANING),
    'no_flip_any': lambda r: not (variant(r) and r['game'] in FLIP_MEANING | FLIP_WIN),
    'no_puzzle': lambda r: r['game'] not in PUZZLE,
}


def game_mean(rows):
    by = defaultdict(list)
    for r in rows:
        by[r['game']].append((r['correct'] - 1 / r['n']) / (1 - 1 / r['n']))
    return statistics.mean(statistics.mean(v) for v in by.values())


def habit_in_errors(rows, conf_cut):
    wrong = [r for r in rows if variant(r) and r['status'] != 'unsupported' and not r['correct']]
    out = {}
    for name, rs in (('confident', [r for r in wrong if r['conf'] >= conf_cut]),
                     ('rest', [r for r in wrong if r['conf'] < conf_cut])):
        out[name] = {'wrong': len(rs), 'habit': sum(r['habit'] for r in rs),
                     'expected_if_random': round(sum(1 / (r['n'] - 1) for r in rs), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', type=Path, default=HERE / 'sets' / 'v1')
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--systems', nargs='+', required=True)
    ap.add_argument('--conf', type=float, default=0.8)
    a = ap.parse_args()
    items = [json.loads(line) for line in (a.set / 'items.jsonl').open()]
    gold = json.loads((a.set / 'gold.json').read_text())
    out = {'cuts': {}, 'ranking': {}, 'habit_in_errors': {}}
    for s in a.systems:
        rows = score_system(items, gold, a.tag, s)
        out['cuts'][s] = {c: round(game_mean([r for r in rows if f(r)]), 3) for c, f in CUTS.items()}
        out['habit_in_errors'][s] = habit_in_errors(rows, a.conf)
    for c in CUTS:
        out['ranking'][c] = sorted(a.systems, key=lambda s: -out['cuts'][s][c])
    out['items'] = {c: sum(1 for it in items if f({**it, 'correct': 0, 'n': 2})) for c, f in CUTS.items()}
    (HERE / 'results' / a.tag / 'audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    for s in a.systems:
        print(s, out['cuts'][s], out['habit_in_errors'][s])
    for c, r in out['ranking'].items():
        print(c, out['items'][c], ' > '.join(r))


if __name__ == '__main__':
    main()
