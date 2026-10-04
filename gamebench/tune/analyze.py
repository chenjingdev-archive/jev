"""판 게임 특화 학습 실험 채점 (tune/PROTOCOL.md 4절). 묶음별 chance-normalized score와 기준선 대비 McNemar.

  .venv/bin/python gamebench/tune/analyze.py
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from score import score_system  # noqa: E402

TRAIN = {'minesweeper', 'sudoku4', 'tictactoe', 'connect4'}
HELD = {'gomoku', 'othello', 'maze', 'snake'}
TUNED, BASE = 'jeff-0.8b-board', 'jeff-0.8b-torch'
SYSTEMS = ['jev-1.13.0', 'kev', 'semif', 'jeff-0.8b', BASE, TUNED]
annotated = lambda r: '주석' in r['stage']
GROUPS = {
    'A 학습 게임': lambda r: r['game'] in TRAIN and not annotated(r),
    'B 새 판 게임': lambda r: r['game'] in HELD and not annotated(r),
    "B' 새 판 게임의 변형": lambda r: r['game'] in HELD and r['stage'].startswith('2'),
    'C (주석) 문항': lambda r: (r['game'] in TRAIN | HELD) and annotated(r),
    'D 판 없는 게임': lambda r: r['game'] not in TRAIN | HELD,
}


def game_mean(rows):
    by = defaultdict(list)
    for r in rows:
        by[r['game']].append((r['correct'] - 1 / r['n']) / (1 - 1 / r['n']))
    return statistics.mean(statistics.mean(v) for v in by.values()) if by else None


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    items = [json.loads(line) for line in (HERE.parent / 'sets/v1/items.jsonl').open()]
    gold = json.loads((HERE.parent / 'sets/v1/gold.json').read_text())
    rows = {s: {r['id']: r for r in score_system(items, gold, 'v1', s)} for s in SYSTEMS}
    out = {}
    for g, f in GROUPS.items():
        ids = [it['id'] for it in items if f(it)]
        o = {'items': len(ids)}
        for s in SYSTEMS:
            rs = [rows[s][i] for i in ids if i in rows[s]]
            o[s] = {'scored': len(rs), 'score': round(game_mean(rs), 3) if rs else None,
                    'accuracy': round(statistics.mean(r['correct'] for r in rs), 3) if rs else None}
        both = [i for i in ids if i in rows[TUNED] and i in rows[BASE]]
        b = sum(rows[TUNED][i]['correct'] and not rows[BASE][i]['correct'] for i in both)
        c = sum(rows[BASE][i]['correct'] and not rows[TUNED][i]['correct'] for i in both)
        o['tuned_vs_base'] = {'only_tuned': b, 'only_base': c, 'mcnemar_p': mcnemar(b, c)}
        out[g] = o
    (HERE / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    for g, o in out.items():
        print(f"{g} ({o['items']}): " + ', '.join(f"{s} {o[s]['score']}" for s in SYSTEMS) + f" | {o['tuned_vs_base']}")


if __name__ == '__main__':
    main()
