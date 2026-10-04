"""사목 한 게임 학습 실험 채점 (c4/PROTOCOL.md 4절). 묶음별 정확도·chance-normalized score와 기준선 대비 McNemar.

  .venv/bin/python gamebench/tune/c4/analyze.py [--tuned jeff-0.8b-c4-1k]
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
from analyze import game_mean, mcnemar  # noqa: E402
from score import score_system  # noqa: E402

BASE, JEV = 'jeff-0.8b-torch', 'jev-1.13.0'
BOARD = {'minesweeper', 'sudoku4', 'tictactoe', 'connect4', 'gomoku', 'othello', 'maze', 'snake'}
annotated = lambda r: '주석' in r['stage']
STAGES = ['1-판정', '2-변형', '3-한수(판)', '4-앞보기(판)']
GROUPS = {  # (tag, 문항 조건)
    'T 사목 시험': ('c4test', lambda r: r['game'] == 'connect4'),
    **{f'T {s}': ('c4test', lambda r, s=s: r['game'] == 'connect4' and r['stage'] == s) for s in STAGES},
    'N 오목 시험': ('c4test', lambda r: r['game'] == 'gomoku'),
    **{f'N {s}': ('c4test', lambda r, s=s: r['game'] == 'gomoku' and r['stage'] == s) for s in STAGES},
    'v1 사목': ('v1', lambda r: r['game'] == 'connect4' and not annotated(r)),
    'v1 틱택토': ('v1', lambda r: r['game'] == 'tictactoe' and not annotated(r)),
    'v1 그 밖의 판 게임': ('v1', lambda r: r['game'] in BOARD - {'connect4', 'tictactoe'} and not annotated(r)),
    'v1 판 없는 게임': ('v1', lambda r: r['game'] not in BOARD),
}
SETS = {'c4test': HERE / 'test', 'v1': HERE.parent.parent / 'sets' / 'v1'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tuned', default='jeff-0.8b-c4-1k')
    a = ap.parse_args()
    systems = [JEV, BASE, a.tuned]
    rows, items = {}, {}
    for tag, d in SETS.items():
        items[tag] = [json.loads(line) for line in (d / 'items.jsonl').open()]
        gold = json.loads((d / 'gold.json').read_text())
        rows[tag] = {s: {r['id']: r for r in score_system(items[tag], gold, tag, s)} for s in systems}
    out = {}
    for g, (tag, f) in GROUPS.items():
        ids = [it['id'] for it in items[tag] if f(it)]
        R = rows[tag]
        o = {'items': len(ids)}
        for s in systems:
            rs = [R[s][i] for i in ids if i in R[s]]
            o[s] = {'scored': len(rs), 'score': round(game_mean(rs), 3) if rs else None,
                    'accuracy': round(statistics.mean(r['correct'] for r in rs), 3) if rs else None}
        both = [i for i in ids if i in R[a.tuned] and i in R[BASE]]
        b = sum(R[a.tuned][i]['correct'] and not R[BASE][i]['correct'] for i in both)
        c = sum(R[BASE][i]['correct'] and not R[a.tuned][i]['correct'] for i in both)
        o['tuned_vs_base'] = {'only_tuned': b, 'only_base': c, 'mcnemar_p': mcnemar(b, c)}
        out[g] = o
    (HERE / f'results-{a.tuned}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    for g, o in out.items():
        print(f"{g} ({o['items']}): " + ', '.join(f"{s} {o[s]['accuracy']}/{o[s]['score']}({o[s]['scored']})" for s in systems)
              + f" | {o['tuned_vs_base']}")


if __name__ == '__main__':
    main()
